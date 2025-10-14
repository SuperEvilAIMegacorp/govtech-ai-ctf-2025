#!/bin/sh

# Best Hedgehog Challenge Docker Entrypoint Script

set -e

echo "🦔 Starting Best Hedgehog Challenge: AI Model Training Data Poisoning"
echo "=================================================="

# Wait for any dependencies (none needed for this stateless challenge)
echo "📋 Challenge Type: Stateless SQL Injection → AI Data Poisoning"
echo "🎯 Goal: Make 'jaga' achieve AI evaluation score of 100"
echo "💡 Each request creates fresh temporary database"

# Set Flask configuration for production
export FLASK_APP=app.py
export FLASK_ENV=production

# Start the Flask application
echo "🚀 Starting Best Hedgehog Flask server on 0.0.0.0:5000..."
exec python app.py