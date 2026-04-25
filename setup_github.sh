#!/bin/bash
# GitHub 远程仓库配置脚本
# 请先在 GitHub 创建仓库，然后运行此脚本

echo "=========================================="
echo "GitHub 远程仓库配置"
echo "=========================================="
echo ""

# 检查当前目录
if [ ! -d ".git" ]; then
    echo "错误: 当前目录不是 Git 仓库"
    exit 1
fi

echo "当前 Git 配置:"
echo "  用户名: $(git config user.name)"
echo "  邮箱: $(git config user.email)"
echo ""

# 提示用户输入 GitHub 用户名
echo "请输入你的 GitHub 用户名 (不是邮箱):"
read -p "GitHub 用户名: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "错误: GitHub 用户名不能为空"
    exit 1
fi

# 构建远程仓库地址
REPO_URL="https://github.com/$GITHUB_USERNAME/enhanceSAM.git"

echo ""
echo "将要配置的远程仓库地址:"
echo "  $REPO_URL"
echo ""

# 检查是否已有远程仓库
if git remote | grep -q "origin"; then
    echo "警告: 已存在名为 'origin' 的远程仓库"
    echo "当前 origin: $(git remote get-url origin)"
    read -p "是否要替换? (y/n): " REPLACE
    if [ "$REPLACE" = "y" ]; then
        git remote remove origin
        echo "已删除旧的 origin"
    else
        echo "取消配置"
        exit 0
    fi
fi

# 添加远程仓库
echo ""
echo "正在添加远程仓库..."
git remote add origin "$REPO_URL"

if [ $? -eq 0 ]; then
    echo "✓ 远程仓库添加成功"
else
    echo "✗ 远程仓库添加失败"
    exit 1
fi

# 重命名分支为 main
echo ""
echo "正在重命名分支为 main..."
git branch -M main

if [ $? -eq 0 ]; then
    echo "✓ 分支重命名成功"
else
    echo "✗ 分支重命名失败"
fi

# 显示当前状态
echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "远程仓库: $(git remote get-url origin)"
echo "当前分支: $(git branch --show-current)"
echo "提交数量: $(git rev-list --count HEAD)"
echo ""
echo "下一步: 推送到 GitHub"
echo "  命令: git push -u origin main"
echo ""
echo "注意: 首次推送需要输入 GitHub 凭据"
echo "  用户名: $GITHUB_USERNAME"
echo "  密码: 使用 Personal Access Token (不是密码)"
echo ""
echo "如何获取 Token:"
echo "  1. 访问 https://github.com/settings/tokens"
echo "  2. 点击 'Generate new token (classic)'"
echo "  3. 勾选 'repo' 权限"
echo "  4. 复制生成的 token"
echo ""
