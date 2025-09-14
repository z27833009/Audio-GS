# Git Setup Guide for Audio-GS

## 步骤 1: 在GitHub创建仓库

1. 打开 https://github.com
2. 点击右上角的 "+" → "New repository"
3. 设置仓库信息：
   - Repository name: `Audio-GS`
   - Description: `Content-Adaptive Audio Representation via 2D Gaussians`
   - Public/Private: 选择你需要的
   - **不要** 勾选 "Initialize this repository with a README"
   - **不要** 添加 .gitignore 或 license
4. 点击 "Create repository"
5. 保持页面打开，稍后需要仓库URL

## 步骤 2: 初始化本地Git仓库

在 F:/Code/Audio-GS 目录下执行：

```bash
# 1. 初始化git仓库
git init

# 2. 添加所有文件
git add .

# 3. 创建首次提交
git commit -m "Initial commit: Audio-GS - 2D Gaussian audio compression"
```

## 步骤 3: 连接到GitHub并推送

```bash
# 1. 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/Audio-GS.git

# 2. 设置主分支名称为main
git branch -M main

# 3. 推送到GitHub
git push -u origin main
```

## 如果使用SSH（推荐）

```bash
# 使用SSH URL（更安全）
git remote add origin git@github.com:YOUR_USERNAME/Audio-GS.git
git branch -M main
git push -u origin main
```

## 完整命令序列

复制粘贴以下命令（记得替换YOUR_USERNAME）：

```bash
cd F:/Code/Audio-GS
git init
git add .
git commit -m "Initial commit: Audio-GS - 2D Gaussian audio compression"
git remote add origin https://github.com/YOUR_USERNAME/Audio-GS.git
git branch -M main
git push -u origin main
```

## 可选：添加项目徽章

推送后，编辑GitHub上的README.md，在顶部添加：

```markdown
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/Audio-GS)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/Audio-GS)
![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/Audio-GS)
```

## 后续更新

```bash
# 添加改动
git add .

# 提交
git commit -m "你的提交信息"

# 推送
git push
```

## 常见问题

### 1. 认证失败
- 使用Personal Access Token代替密码
- 或配置SSH密钥

### 2. 大文件警告
- .gitignore已配置忽略音频文件
- 如需上传示例音频，使用Git LFS

### 3. 更改远程URL
```bash
git remote set-url origin 新的URL
```

## Git LFS（处理大文件）

如果要上传音频示例：

```bash
# 安装Git LFS
git lfs install

# 跟踪音频文件
git lfs track "*.wav"
git lfs track "*.mp3"

# 添加.gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```