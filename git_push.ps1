# PowerShell script to initialize and push to GitHub
# 使用前请修改YOUR_USERNAME为你的GitHub用户名

$username = Read-Host "Enter your GitHub username"
$repoUrl = "https://github.com/$username/Audio-GS.git"

Write-Host "Initializing Git repository..." -ForegroundColor Green
git init

Write-Host "`nAdding files..." -ForegroundColor Green
git add .

Write-Host "`nCreating initial commit..." -ForegroundColor Green
git commit -m "Initial commit: Audio-GS - 2D Gaussian audio compression

- Core model implementation with 2D Gaussians in time-frequency domain
- Support for speech and music compression
- Configurable quality-bitrate tradeoff
- Quantization support for reduced file size
- Multiple training configurations
- Comprehensive documentation in English and Chinese"

Write-Host "`nAdding remote repository: $repoUrl" -ForegroundColor Green
git remote add origin $repoUrl

Write-Host "`nSetting main branch..." -ForegroundColor Green
git branch -M main

Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`nDone! Your repository is now on GitHub:" -ForegroundColor Green
Write-Host "https://github.com/$username/Audio-GS" -ForegroundColor Cyan