@echo off
echo ========================================
echo Git Push Script for Audio-GS
echo ========================================
echo.

set /p USERNAME="Enter your GitHub username: "
set REPO_URL=https://github.com/%USERNAME%/Audio-GS.git

echo.
echo Initializing Git repository...
git init

echo.
echo Adding all files...
git add .

echo.
echo Creating initial commit...
git commit -m "Initial commit: Audio-GS - 2D Gaussian audio compression"

echo.
echo Adding remote repository: %REPO_URL%
git remote add origin %REPO_URL%

echo.
echo Setting main branch...
git branch -M main

echo.
echo Pushing to GitHub...
echo Note: You may need to enter your GitHub credentials
git push -u origin main

echo.
echo ========================================
echo Done! Your repository is now on GitHub:
echo https://github.com/%USERNAME%/Audio-GS
echo ========================================
echo.
pause