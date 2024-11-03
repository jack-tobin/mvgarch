#!/bin/bash

# Extract current version from setup.py
current_version=$(grep -m1 'VERSION = ' setup.py | cut -d'"' -f2)
echo "Current version: $current_version"

# Split version into components
major=$(echo "$current_version" | cut -d. -f1)
minor=$(echo "$current_version" | cut -d. -f2)
patch=$(echo "$current_version" | cut -d. -f3)
new_patch=$((patch + 1))
new_version="${major}.${minor}.${new_patch}"
echo "New version: $new_version"

# Update setup.py with new version
sed -i "s/VERSION = \".*\"/VERSION = \"$new_version\"/" setup.py

# Configure git
git config --local user.email "github-actions[bot]@users.noreply.github.com"
git config --local user.name "github-actions[bot]"

# Commit and push changes
git add setup.py
git commit -m "Bump version to $new_version"
git push
