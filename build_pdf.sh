#!/bin/bash

# Script to build LaTeX PDF and clean up artifacts
# Usage: ./build_pdf.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARTICLE_DIR="${SCRIPT_DIR}/article"

echo -e "${GREEN}Building LaTeX document...${NC}"

# Change to article directory
cd "$ARTICLE_DIR"

# Check if main.tex exists
if [ ! -f "main.tex" ]; then
    echo -e "${RED}Error: main.tex not found in ${ARTICLE_DIR}${NC}"
    exit 1
fi

# Compile the document
echo -e "${YELLOW}Compiling LaTeX document...${NC}"
if latexmk -pdf -interaction=nonstopmode main.tex; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi

# Clean up artifacts
echo -e "${YELLOW}Cleaning up artifacts...${NC}"

# Remove common LaTeX auxiliary files
rm -f main.aux
rm -f main.log
rm -f main.bbl
rm -f main.bcf
rm -f main.blg
rm -f main.fls
rm -f main.fdb_latexmk
rm -f main.out
rm -f main.run.xml
rm -f main.synctex.gz
rm -f main.toc
rm -f main.lof
rm -f main.lot
rm -f main.nav
rm -f main.snm
rm -f main.vrb
rm -f main.spl

# Remove latexmk-specific files
rm -f main.synctex
rm -f main.synctex\(busy\)

echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo -e "${GREEN}PDF saved at: ${ARTICLE_DIR}/main.pdf${NC}"

