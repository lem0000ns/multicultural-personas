#!/bin/bash

# Script to run the BLEnD Results Viewer

echo "🚀 Starting BLEnD Results Viewer..."
echo ""
echo "📊 The viewer will automatically open in your browser"
echo "If it doesn't, navigate to http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
streamlit run blend_viewer.py
