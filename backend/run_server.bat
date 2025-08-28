@echo off
set SPOTIPY_CLIENT_ID="8dd72a61559d4d06835097ba58c716b3"
set SPOTIPY_CLIENT_SECRET="bb57676e5c9347dba1d99e440dd173ba"
set SPOTIPY_REDIRECT_URI="http://127.0.0.1:8888/callback"

echo Starting the server...
uvicorn server:app --reload
pause