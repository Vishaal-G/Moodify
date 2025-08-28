import os
import subprocess

# ====================================================================
# CRITICAL: PLACE YOUR SPOTIFY CREDENTIALS HERE
# ====================================================================
# These will be set as environment variables for the server process.
CLIENT_ID = "8dd72a61559d4d06835097ba58c716b3"
CLIENT_SECRET = "bb57676e5c9347dba1d99e440dd173ba"

# Check for placeholders before starting the server
if CLIENT_ID == "<YOUR_SPOTIFY_CLIENT_ID>" or CLIENT_SECRET == "<YOUR_SPOTIFY_CLIENT_SECRET>":
    print("ERROR: Please replace the placeholder values in start_server.py with your Spotify credentials.")
    exit()

# Set the environment variables
os.environ['SPOTIPY_CLIENT_ID'] = CLIENT_ID
os.environ['SPOTIPY_CLIENT_SECRET'] = CLIENT_SECRET
os.environ['UVICORN_PORT'] = "8000"

print("Starting server with Uvicorn...")
print("-" * 30)

try:
    # Run the uvicorn server as a subprocess
    process = subprocess.Popen(
        [
            "uvicorn",
            "server:app",
            "--host", "0.0.0.0",
            "--port", os.environ.get("UVICORN_PORT"),
            "--reload",
        ],
        env=os.environ,
    )
    process.wait()

except FileNotFoundError:
    print("ERROR: Uvicorn not found. Please make sure it is installed.")
    print("You can install it by running 'pip install uvicorn'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")