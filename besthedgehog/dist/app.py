#!/usr/bin/env python3
"""
Best Hedgehog Evaluation System

An AI-powered system for evaluating hedgehog performance based on qualities.
The system trains a neural network on hedgehog data to predict overall evaluation scores.

This application is STATELESS - each session creates its own temporary database.
"""

from flask import Flask, render_template, request, jsonify, session
import sqlite3
import torch
import torch.nn as nn
import numpy as np
from model import HedgehogEvaluationModel, train_model, evaluate_hedgehog
import os
import uuid
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
FLAG = os.getenv('FLAG', 'flag{default_flag_not_configured}')

# Validate that the flag is properly configured
if FLAG == 'flag{default_flag_not_configured}':
    print("⚠️  WARNING: FLAG not configured! Please set FLAG in .env file")
    print("💡 Copy .env.example to .env and set your flag")

def create_temp_database():
    """Create a temporary in-memory database with sample data."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create the hedgehogs table
    cursor.execute('''
        CREATE TABLE hedgehogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            furriness INTEGER DEFAULT 0,
            cuteness INTEGER DEFAULT 0,
            friendliness INTEGER DEFAULT 0,
            curiosity INTEGER DEFAULT 0,
            agility INTEGER DEFAULT 0,
            sleepiness INTEGER DEFAULT 0,
            evaluation_score REAL DEFAULT NULL
        )
    ''')
    
    # Insert sample data with predefined target evaluation scores
    # Format: (username, furriness, cuteness, friendliness, curiosity, agility, sleepiness, target_evaluation_score)
    hedgehog_data = [
        ('alice', 85, 92, 78, 65, 88, 79, 85.5),      # Fluffy and cute
        ('bob', 76, 84, 89, 91, 73, 82, 82.7),        # Friendly and curious
        ('charlie', 94, 67, 85, 78, 92, 88, 84.0),    # Very furry and agile
        ('diana', 88, 95, 71, 83, 76, 90, 85.2),      # Super cute and sleepy
        ('eve', 79, 88, 94, 69, 84, 75, 81.5),        # Very friendly
        ('frank', 91, 73, 82, 95, 79, 86, 84.3),      # Very curious
        ('grace', 83, 89, 77, 87, 91, 74, 83.5),      # Agile and cute
        ('henry', 87, 76, 93, 74, 85, 89, 84.0),      # Friendly and sleepy
        ('iris', 74, 91, 68, 92, 88, 93, 84.3),       # Cute, curious, sleepy
        ('jack', 90, 82, 86, 80, 77, 91, 84.3),       # Balanced qualities
        ('jaga', 45, 50, 40, 35, 48, 42, None)        # Target hedgehog - no target score
    ]
    
    # Extract training data with target scores (all hedgehogs except jaga)
    training_data = []
    training_targets = []
    
    for hedgehog in hedgehog_data:
        if hedgehog[0] != 'jaga':  # Skip jaga for training
            scores = [hedgehog[1], hedgehog[2], hedgehog[3], hedgehog[4], hedgehog[5], hedgehog[6]]
            target_score = hedgehog[7]
            
            training_data.append(scores)
            training_targets.append(target_score)
    
    # Ensure we have training data
    if len(training_data) == 0:
        raise RuntimeError("No training data available for baseline model")
    
    if len(training_targets) != len(training_data):
        raise RuntimeError("Training data and target scores must have same length")
    
    # Train baseline model with predefined target evaluation scores
    try:
        baseline_model = train_model(training_data, training_targets)
        if baseline_model is None:
            raise RuntimeError("Model training failed - returned None")
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")
    
    # Prepare sample data with predefined target evaluation scores
    sample_hedgehogs = []
    for hedgehog in hedgehog_data:
        username = hedgehog[0]
        target_evaluation = hedgehog[7]  # Predefined target evaluation score for training data
        
        if username == 'jaga':
            # Jaga's evaluation score is calculated dynamically (test data)
            stored_evaluation_score = None
        else:
            # Use predefined target evaluation score for training data
            stored_evaluation_score = float(target_evaluation)

        sample_hedgehogs.append(
            (
                username,
                hedgehog[1], hedgehog[2], hedgehog[3],
                hedgehog[4], hedgehog[5], hedgehog[6],
                stored_evaluation_score,
            )
        )
    
    # Insert hedgehogs into database
    cursor.executemany('''
        INSERT INTO hedgehogs (username, furriness, cuteness, friendliness, 
                               curiosity, agility, sleepiness, evaluation_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_hedgehogs)
    
    conn.commit()
    
    # Verify the evaluation scores were calculated and stored correctly
    cursor.execute("SELECT COUNT(*) FROM hedgehogs WHERE evaluation_score IS NOT NULL")
    training_count = cursor.fetchone()[0]
    
    if training_count == 0:
        raise RuntimeError("Failed to calculate and store evaluation scores for training data")
    
    return conn

@app.route('/')
def index():
    """Main page with fresh database for each session."""
    # Create fresh temporary database
    conn = create_temp_database()
    hedgehogs = conn.execute('SELECT * FROM hedgehogs ORDER BY username').fetchall()
    
    # Store database data in session for this request cycle
    hedgehogs_list = []
    for hh in hedgehogs:
        hedgehogs_list.append({
            'id': hh['id'],
            'username': hh['username'],
            'furriness': hh['furriness'],
            'cuteness': hh['cuteness'],
            'friendliness': hh['friendliness'],
            'curiosity': hh['curiosity'],
            'agility': hh['agility'],
            'sleepiness': hh['sleepiness'],
            'evaluation_score': hh['evaluation_score']
        })
    
    conn.close()
    return render_template('index.html', hedgehogs=hedgehogs_list)

@app.route('/add_hedgehog', methods=['POST'])
def add_hedgehog():
    """Add a new hedgehog to the database and retrain the AI model."""
    username = request.form['username']
    furriness = request.form['furriness']
    cuteness = request.form['cuteness']
    friendliness = request.form['friendliness']
    curiosity = request.form['curiosity']
    agility = request.form['agility']
    sleepiness = request.form['sleepiness']
    
    # Get optional evaluation score
    evaluation_score = request.form.get('evaluation_score')
    if evaluation_score and evaluation_score.strip():
        try:
            evaluation_score = float(evaluation_score)
            if evaluation_score < 0 or evaluation_score > 100:
                return jsonify({
                    'success': False, 
                    'message': 'Evaluation score must be between 0 and 100.',
                    'hedgehogs': []
                })
        except ValueError:
            return jsonify({
                'success': False, 
                'message': 'Invalid evaluation score format.',
                'hedgehogs': []
            })
    else:
        evaluation_score = None
    
    # Create fresh temporary database for this attempt
    conn = create_temp_database()
    
    # Build query to insert new hedgehog
    if evaluation_score is not None:
        query = f"""
            INSERT INTO hedgehogs (username, furriness, cuteness, friendliness, 
                                   curiosity, agility, sleepiness, evaluation_score)
            VALUES ('{username}', {furriness}, {cuteness}, {friendliness}, 
                    {curiosity}, {agility}, {sleepiness}, {evaluation_score})
        """
    else:
        query = f"""
            INSERT INTO hedgehogs (username, furriness, cuteness, friendliness, 
                                   curiosity, agility, sleepiness)
            VALUES ('{username}', {furriness}, {cuteness}, {friendliness}, 
                    {curiosity}, {agility}, {sleepiness})
        """
    
    if "jaga" in query:
        return jsonify({
            'success': False, 
            'message': f'Failed to execute query: Attempt to modify jaga is not allowed.',
            'hedgehogs': []
        })

    try:
        # Execute the potentially malicious query
        conn.executescript(query)  # Using executescript to allow multiple statements
        conn.commit()
        
        # Get all hedgehogs after adding new entry
        hedgehogs = conn.execute('SELECT * FROM hedgehogs ORDER BY username').fetchall()
        
        # Separate training data (all hedgehogs EXCEPT jaga) and test data (jaga)
        training_data = []
        training_targets = []
        jaga_data = None
        
        for hedgehog in hedgehogs:
            scores = [
                hedgehog['furriness'], hedgehog['cuteness'], hedgehog['friendliness'],
                hedgehog['curiosity'], hedgehog['agility'], hedgehog['sleepiness']
            ]

            if hedgehog['username'] == 'jaga':
                jaga_data = hedgehog
            else:
                # Add all other hedgehogs to training data (excluding jaga)
                training_data.append(scores)
                # Use stored evaluation score as target (if available)
                if hedgehog['evaluation_score'] is not None:
                    training_targets.append(float(hedgehog['evaluation_score']))
                else:
                    # Fallback to synthetic target if no stored score
                    avg_score = sum(scores) / len(scores)
                    training_targets.append(avg_score)
        
        if not jaga_data:
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'Jaga not found in database!',
                'hedgehogs': []
            })
        
        if len(training_data) == 0:
            conn.close()
            return jsonify({
                'success': False, 
                'message': 'No training data available (need hedgehogs other than jaga)!',
                'hedgehogs': []
            })
        
        # Train the model ONLY on non-jaga hedgehogs using their stored evaluation scores as targets
        model = train_model(training_data, training_targets)
        
        # Test the model on jaga (jaga is the test data, not training data)
        jaga_scores = [
            jaga_data['furriness'], jaga_data['cuteness'], jaga_data['friendliness'],
            jaga_data['curiosity'], jaga_data['agility'], jaga_data['sleepiness']
        ]
        
        jaga_evaluation = evaluate_hedgehog(model, jaga_scores)
            
        # Convert hedgehogs to list using stored evaluation scores for training data
        hedgehogs_list = []
        training_evaluations = []

        for hedgehog in hedgehogs:
            if hedgehog['username'] == 'jaga':
                hedgehog_evaluation = jaga_evaluation
            else:
                # For training data, use model evaluation on current scores
                hedgehog_scores = [
                    hedgehog['furriness'], hedgehog['cuteness'], hedgehog['friendliness'],
                    hedgehog['curiosity'], hedgehog['agility'], hedgehog['sleepiness']
                ]
                hedgehog_evaluation = evaluate_hedgehog(model, hedgehog_scores)

            hedgehog_data = {
                'id': hedgehog['id'],
                'username': hedgehog['username'],
                'furriness': hedgehog['furriness'],
                'cuteness': hedgehog['cuteness'],
                'friendliness': hedgehog['friendliness'],
                'curiosity': hedgehog['curiosity'],
                'agility': hedgehog['agility'],
                'sleepiness': hedgehog['sleepiness'],
                'evaluation_score': float(hedgehog_evaluation),
                'stored_evaluation_score': float(hedgehog['evaluation_score']) if hedgehog['evaluation_score'] is not None else None,
                'is_training_data': hedgehog['username'] != 'jaga'
            }

            hedgehogs_list.append(hedgehog_data)

            # Track training evaluations separately
            if hedgehog['username'] != 'jaga':
                training_evaluations.append({
                    'username': hedgehog['username'],
                    'evaluation_score': float(hedgehog_evaluation),
                    'stored_evaluation_score': float(hedgehog['evaluation_score']) if hedgehog['evaluation_score'] is not None else None
                })
        
        # Prepare response
        training_count = len(training_data)
        response = {
            'success': True,
            'message': f"Hedgehog added! Model trained on {training_count} hedgehogs (excluding jaga). Jaga's test evaluation: {jaga_evaluation:.2f}",
            'jaga_evaluation': float(jaga_evaluation),
            'jaga_scores': jaga_scores,
            'hedgehogs': hedgehogs_list,
            'training_count': training_count,
            'training_evaluations': training_evaluations
        }
        
        # Check for flag
        if jaga_evaluation >= 100.0:
            response['flag'] = FLAG
            response['flag_found'] = True
            response['message'] += f" 🎉 FLAG CAPTURED: {FLAG}"
        else:
            response['flag_found'] = False
            response['message'] += " (Need 100.0 to get the flag!)"
        
        conn.close()
        return jsonify(response)
        
    except Exception as e:
        conn.close()
        return jsonify({
            'success': False, 
            'message': f'Failed to add hedgehog: {str(e)}',
            'hedgehogs': []
        })

@app.route('/demo', methods=['POST'])
def demo_clean_training():
    """Demonstrate training on fresh database with default data."""
    try:
        # Create fresh database
        conn = create_temp_database()
        hedgehogs = conn.execute('SELECT * FROM hedgehogs').fetchall()
        
        # Separate training data (all hedgehogs EXCEPT jaga) and test data (jaga)
        training_data = []
        training_targets = []
        jaga_data = None
        
        for hedgehog in hedgehogs:
            scores = [
                hedgehog['furriness'], hedgehog['cuteness'], hedgehog['friendliness'],
                hedgehog['curiosity'], hedgehog['agility'], hedgehog['sleepiness']
            ]

            if hedgehog['username'] == 'jaga':
                jaga_data = hedgehog
            else:
                # Add all other hedgehogs to training data (excluding jaga)
                training_data.append(scores)
                # Use stored evaluation score as target (if available)
                if hedgehog['evaluation_score'] is not None:
                    training_targets.append(float(hedgehog['evaluation_score']))
                else:
                    # Fallback to synthetic target if no stored score
                    avg_score = sum(scores) / len(scores)
                    training_targets.append(avg_score)
        
        if not jaga_data:
            conn.close()
            return jsonify({'success': False, 'message': 'Jaga not found in database!'})
        
        # Train the model ONLY on non-jaga hedgehogs using their stored evaluation scores as targets
        model = train_model(training_data, training_targets)
        
        # Test the model on jaga (jaga is test data, not training data)
        jaga_scores = [
            jaga_data['furriness'], jaga_data['cuteness'], jaga_data['friendliness'],
            jaga_data['curiosity'], jaga_data['agility'], jaga_data['sleepiness']
        ]
        
        jaga_evaluation = evaluate_hedgehog(model, jaga_scores)
        
        # Convert hedgehogs to list using stored evaluation scores for training data
        hedgehogs_list = []
        training_evaluations = []

        for hedgehog in hedgehogs:
            if hedgehog['username'] == 'jaga':
                # For jaga, use calculated evaluation score (test data)
                hedgehog_evaluation = jaga_evaluation
            else:
                # For training data, use stored evaluation score from database
                hedgehog_evaluation = hedgehog['evaluation_score'] if hedgehog['evaluation_score'] is not None else 0.0

            hedgehog_data = {
                'id': hedgehog['id'],
                'username': hedgehog['username'],
                'furriness': hedgehog['furriness'],
                'cuteness': hedgehog['cuteness'],
                'friendliness': hedgehog['friendliness'],
                'curiosity': hedgehog['curiosity'],
                'agility': hedgehog['agility'],
                'sleepiness': hedgehog['sleepiness'],
                'evaluation_score': float(hedgehog_evaluation),
                'stored_evaluation_score': float(hedgehog['evaluation_score']) if hedgehog['evaluation_score'] is not None else None,
                'is_training_data': hedgehog['username'] != 'jaga'
            }

            hedgehogs_list.append(hedgehog_data)

            # Track training evaluations separately
            if hedgehog['username'] != 'jaga':
                training_evaluations.append({
                    'username': hedgehog['username'],
                    'evaluation_score': float(hedgehog_evaluation),
                    'stored_evaluation_score': float(hedgehog['evaluation_score']) if hedgehog['evaluation_score'] is not None else None
                })
        
        conn.close()
        
        training_count = len(training_data)
        response = {
            'success': True,
            'jaga_evaluation': float(jaga_evaluation),
            'jaga_scores': jaga_scores,
            'message': f"Clean training on {training_count} hedgehogs (excluding jaga). Jaga's test evaluation: {jaga_evaluation:.2f}",
            'hedgehogs': hedgehogs_list,
            'training_count': training_count,
            'training_evaluations': training_evaluations
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Demo training failed: {str(e)}'})

if __name__ == '__main__':
    print("🦔 Best Hedgehog Evaluation System")
    print("📍 Server starting at http://localhost:5000")
    print("🤖 AI-powered system for evaluating hedgehog performance")
    print("💡 Each request creates a fresh temporary database")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
