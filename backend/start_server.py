import os
import subprocess
from dotenv import load_dotenv

# ====================================================================
# Load environment variables from .env file
# ====================================================================
# The `dotenv` library will automatically load the key-value pairs from
# the .env file into the system's environment variables.
load_dotenv()

# ====================================================================
# CRITICAL: DO NOT PLACE YOUR SPOTIFY CREDENTIALS HERE
# ====================================================================
# Instead, they are now loaded from the .env file.
# We retrieve them using `os.getenv()`.
CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# ====================================================================
# Check for credentials before starting the server
# ====================================================================
if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: Spotify credentials not found.")
    print("Please ensure you have created a '.env' file in the root directory")
    print("and set the SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET variables.")
    exit()

# Set the port for Uvicorn
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
        env=os.environ.copy()
    )

    process.wait()

except KeyboardInterrupt:
    print("\nServer process terminated by user.")
except Exception as e:
    print(f"An error occurred: {e}")

