import os
import random
import time
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Config
# =========================
# Set to 'None' to use default from model
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Mapping of emotions to a broad set of search keywords.
# This list is now used to build a search query for the sp.search() method.
EMOTION_KEYWORDS = {
    'anger':    ['angry', 'rebellious', 'aggressive', 'metal', 'hardcore'],
    'disgust':  ['disturbing', 'grindcore', 'gothic'],
    'fear':     ['scary', 'spooky', 'dark ambient', 'soundtrack'],
    'joy':      ['happy', 'upbeat', 'celebration', 'pop', 'dance'],
    'sadness':  ['sad', 'melancholy', 'chill', 'acoustic'],
    'surprise': ['unexpected', 'intense', 'electronic', 'trance']
}

# Popularity thresholds (tweak as needed)
MIN_TRACK_POPULARITY_DEFAULT = 60         # 0..100


# =========================
# Small helpers
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamps a value between a lower and upper bound."""
    return max(lo, min(hi, x))

def _normalize_text(s: str) -> str:
    """Normalizes text by converting to lowercase and removing extra whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _get_api_credentials() -> SpotifyClientCredentials:
    """
    Retrieves Spotify credentials from environment variables.
    """
    client_id = os.environ.get('SPOTIPY_CLIENT_ID')
    client_secret = os.environ.get('SPOTIPY_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("ERROR: SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables must be set.")
        raise RuntimeError("Please set the environment variables.")

    print(f"DEBUG: Successfully read credentials from environment.")
    
    return SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

@lru_cache(maxsize=1)
def _get_sp() -> spotipy.Spotify:
    """Creates and caches a Spotipy client instance."""
    auth_manager = _get_api_credentials()
    return spotipy.Spotify(auth_manager=auth_manager)


def _test_spotify_credentials() -> bool:
    """
    Makes a simple API call to verify if credentials are valid.
    Returns True if successful, False otherwise.
    """
    try:
        sp = _get_sp()
        # Make a simple authenticated call
        test_result = sp.search(q="The Beatles", limit=1, type="artist")
        if test_result and 'artists' in test_result:
            print("DEBUG: Spotify credentials test successful. Authentication is working.")
            return True
        print("DEBUG: Spotify credentials test failed: search returned an unexpected result.")
        return False
    except SpotifyException as e:
        print(f"ERROR: Spotify credentials test failed. Received API error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Spotify credentials test: {e}")
        return False


# =========================
# Main recommendation logic
# =========================
def search_for_emotion(
    emotion: str,
    limit: int = 25,
    user_genres: Optional[List[str]] = None,
    min_track_popularity: int = MIN_TRACK_POPULARITY_DEFAULT,
    market: str = 'US',
) -> List[Dict]:
    """
    Core function to get tracks based on emotion and user-defined genres using Spotify's search engine.
    This replaces the now-deprecated recommendations endpoint.
    """
    try:
        sp = _get_sp()
    except RuntimeError as e:
        print(f"ERROR: Failed to get Spotify client: {e}")
        return []

    print(f"DEBUG: Using sp.search() for emotion: '{emotion}'")
    
    tracks = []
    
    # Get search keywords for the predicted emotion
    emotion_keywords = EMOTION_KEYWORDS.get(emotion, [])
    
    # Construct the search query
    query = ""
    # Use the first keyword for the emotion as the base of the search
    if emotion_keywords:
        query = random.choice(emotion_keywords)
        print(f"DEBUG: Using random emotion keyword '{query}' as a base.")
    
    # Add the user's genre to the query if provided
    if user_genres and user_genres[0]:
        query += f" genre:{user_genres[0]}"
        print(f"DEBUG: Added user genre '{user_genres[0]}' to the query.")
    
    if not query:
        print("DEBUG: No keywords found to create a search query. Returning empty list.")
        return []

    print(f"DEBUG: Searching with query: '{query}'")

    try:
        results = sp.search(
            q=query,
            limit=limit,
            type='track',
            market=market
        )
        tracks = results.get('tracks', {}).get('items', [])
        
        # Filter tracks by popularity
        tracks = [t for t in tracks if t.get('popularity', 0) >= min_track_popularity]
        
        print(f"DEBUG: Found {len(tracks)} tracks via search after filtering by popularity.")
    except Exception as e:
        print(f"ERROR: Failed to get tracks via search: {e}")
        tracks = []
        
    return tracks


# =========================
# Model loader
# =========================
@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    """Loads and caches the Hugging Face model and tokenizer."""
    try:
        if MODEL_NAME is None:
            model_name_to_load = "j-hartmann/emotion-english-distilroberta-base"
        else:
            model_name_to_load = MODEL_NAME
            
        tokenizer = AutoTokenizer.from_pretrained(model_name_to_load)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load)
        
        print(f"Model '{model_name_to_load}' loaded successfully.")
        
        id2label = model.config.id2label
        label2id = model.config.label2id
        
        return tokenizer, model, id2label, label2id
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")


# =========================
# API setup
# =========================
app = FastAPI()

# Add CORS middleware to allow cross-origin requests from the frontend
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, tokenizer, and labels once at startup
try:
    tokenizer, model, id2label, label2id = _load_model_and_tokenizer()
    # Perform the credentials check after the model is loaded.
    _test_spotify_credentials()
except RuntimeError as e:
    print(e)
    tokenizer, model, id2label, label2id = None, None, None, None


# =========================
# Pydantic models for API
# =========================
class PredictRequest(BaseModel):
    text: str
    genre: Optional[str] = None

class TrackResponse(BaseModel):
    title: str
    artist: str
    album: str
    cover_art: str
    spotify_url: str

class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    tracks: List[TrackResponse]


# =========================
# API endpoints
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(
    inp: PredictRequest,
    limit: int = Query(25, ge=1, le=50, description="Max tracks to return"),
    min_track_popularity: int = Query(MIN_TRACK_POPULARITY_DEFAULT, ge=0, le=100),
    market: str = Query('US', min_length=2, max_length=2, description="ISO 2-letter market, e.g. US, CA"),
):
    """
    Endpoint to predict emotion from text and recommend music.
    """
    print("DEBUG: Entered the /predict endpoint.")
    print(f"DEBUG: Received text: {inp.text}")
    print(f"DEBUG: Received genre: {inp.genre}")
    print(f"DEBUG: Received limit: {limit}")
    
    if not all([tokenizer, model, id2label, label2id]):
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Check server logs for details."
        )

    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    # Run model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        pred_label = id2label[pred_id]
        conf = float(probs[pred_id])

    print(f"DEBUG: Predicted emotion: {pred_label} with confidence: {conf}")

    # parse/normalize user genres
    def _sanitize_user_genres(g: str) -> List[str]:
        if not g:
            return None
        return [s.strip() for s in g.lower().split(',') if s.strip()]

    user_genres = _sanitize_user_genres(inp.genre) if inp.genre else None
    print(f"DEBUG: Processed genres: {user_genres}")

    # Use the new search function to get tracks
    tracks = search_for_emotion(
        emotion=pred_label,
        limit=limit,
        user_genres=user_genres,
        min_track_popularity=min_track_popularity,
        market=market,
    )

    print(f"DEBUG: Found {len(tracks)} tracks")

    return {
        "emotion": pred_label,
        "confidence": conf,
        "tracks": [
            {
                "title": t.get('name', ''),
                "artist": ", ".join([a.get('name', '') for a in t.get('artists', [])]),
                "album": t.get('album', {}).get('name', ''),
                "cover_art": t.get('album', {}).get('images', [{}])[0].get('url', None),
                "spotify_url": t.get('external_urls', {}).get('spotify', '')
            }
            for t in tracks
        ]
    }

# Also add this for testing
@app.get("/test")
def test_endpoint():
    return {"status": "API is working", "timestamp": time.time()}

