# 🎵 Mood → Music Generator

A full-stack web application that uses an AI model to analyze your mood and generates a custom playlist of songs from Spotify to match your vibe.


<img width="603" height="560" alt="image" src="https://github.com/user-attachments/assets/0416e040-5ef7-4e34-ae97-78230939c536" />


---

## ## Features

✨ **AI-Powered Mood Analysis**: Leverages a local `Transformers` model (PyTorch) to detect emotions like joy, sadness, and anger from your text input.

🎧 **Dynamic Spotify Playlists**: Connects to the Spotify API to find and recommend real songs based on the detected emotion and your preferences.

🎛️ **Customizable Recommendations**: Filter your playlist by music genre and select the exact number of tracks you want (from 1 to 50).

📈 **Popular Artists Filter**: The backend is configured to prioritize songs from artists with over 8 million Spotify followers, ensuring you get well-known tracks.

▶️ **Interactive UI**: A clean and simple interface that includes song previews, links to open tracks directly in Spotify, and dynamic color themes that change based on the generated mood.

⏳ **Smooth Loading States**: Features skeleton loaders and status updates to provide a great user experience while the AI and API do their work.

---

## ## Tech Stack

This project is built with a powerful but lightweight stack:

* **Backend**:
    * **Framework**: FastAPI
    * **AI**: PyTorch & Hugging Face Transformers
    * **Spotify API Client**: Spotipy
    * **Server**: Uvicorn
    * **Environment**: Python 3.8+

* **Frontend**:
    * **Languages**: HTML5, CSS3, Vanilla JavaScript (no frameworks)
    * **API Communication**: Fetch API

---

## ## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

* **Python**: Version 3.8 or higher.
* **Spotify Developer Account**: You need a free developer account to get API keys.
    1.  Go to the [Spotify Developer Dashboard](http://googleusercontent.com/spotify.com/6).
    2.  Click "Create app".
    3.  Note down your **Client ID** and **Client Secret**.

### 1. Backend Setup

First, let's get the Python server running.

```bash
# 1. Clone this repository to your local machine
git clone <your-repository-url>
cd <your-repository-folder>

# 2. Create and activate a Python virtual environment
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
.\venv\Scripts\activate

# 3. Install the required Python packages
pip install "fastapi[all]" torch transformers spotipy python-dotenv

# 4. Create the .env file for your API keys (see Configuration below)

# 5. Run the server!
uvicorn server:app --reload
```

The backend server will now be running at `http://127.0.0.1:8000`.

### 2. Frontend Setup

The frontend is simple and requires no build steps.

1.  After starting the backend server, simply **open the `popup.html` file** in your favorite web browser (like Chrome, Firefox, or Edge).

---

## ## Configuration

You must provide your Spotify API keys for the application to work.

1.  In the main project folder (the same directory as `server.py`), create a new file named `.env`.
2.  Open the `.env` file and add your credentials in the following format. **Do not use quotes.**

    ```env
    SPOTIFY_CLIENT_ID=your_client_id_from_the_spotify_dashboard
    SPOTIFY_CLIENT_SECRET=your_client_secret_from_the_spotify_dashboard
    ```

3.  Save the file. Remember to **restart the Uvicorn server** if it was already running to ensure it loads the new credentials.

---

## ## How to Use

🚀 Once the backend is running and you've opened `popup.html`:

1.  **Type your mood**: Write a sentence or two in the text box describing how you feel.
2.  **Select a Genre**: (Optional) Choose a specific music genre from the dropdown.
3.  **Choose Track Count**: Use the slider to select how many songs you want.
4.  **Generate**: Click the "Generate Songs" button.
5.  **Enjoy**: Listen to song previews directly in the app or click the Spotify icon to open the full track on their platform.
