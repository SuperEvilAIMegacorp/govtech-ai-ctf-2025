// Best Hedgehog Evaluation System JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const hedgehogForm = document.getElementById('hedgehogForm');
    const demoBtn = document.getElementById('demoBtn');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');

    // Handle form submission
    hedgehogForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(hedgehogForm);
        const submitBtn = hedgehogForm.querySelector('button[type="submit"]');
        
        // Validate form
        if (!hedgehogForm.checkValidity()) {
            showNotification('Please fill out all form fields correctly!', 'error');
            return;
        }
        
    submitBtn.disabled = true;
    submitBtn.textContent = 'Adding Hedgehog & Training...';
        submitBtn.classList.add('loading');
        
        const resultDiv = document.getElementById('resultDiv');
        if (resultDiv) resultDiv.style.display = 'none';
        
    fetch('/add_hedgehog', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                let resultClass = 'result-success';
                let resultMessage = data.message;
                
                if (data.flag) {
                    resultClass = 'result-flag';
                    resultMessage = `🎉 CONGRATULATIONS! ${data.message}`;
                    showCelebration();
                }
                
                showResult(resultMessage, resultClass);
                displayResults(data);
                
            } else {
                showResult(data.message, 'result-error');
            }
        })
        .catch(error => {
            showResult('Hedgehog addition failed: ' + error.message, 'result-error');
        })
        .finally(() => {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Add Hedgehog & Train Model';
            submitBtn.classList.remove('loading');
        });
    });

    // Handle demo training
    demoBtn.addEventListener('click', function() {
        demoBtn.disabled = true;
        demoBtn.textContent = 'Running Demo...';
        demoBtn.classList.add('loading');
        
        fetch('/demo', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showResult(`Demo: ${data.message}`, 'result-success');
                displayResults(data);
            } else {
                showResult('Demo failed: ' + data.message, 'result-error');
            }
        })
        .catch(error => {
            showResult('Demo failed: ' + error.message, 'result-error');
        })
        .finally(() => {
            demoBtn.disabled = false;
            demoBtn.textContent = 'Demo Hedgehog Training';
            demoBtn.classList.remove('loading');
        });
    });

    // Show result
    function showResult(message, className) {
        const resultDiv = document.getElementById('resultDiv');
        if (resultDiv) {
            resultDiv.textContent = message;
            resultDiv.className = 'result-box ' + className;
            resultDiv.style.display = 'block';
        }
    }

    // Display detailed results
    function displayResults(data) {
    if (!data.hedgehogs || data.hedgehogs.length === 0) {
            return;
        }

        let html = '<h3>AI Training Results</h3>';
        
        // Show jaga's current scores
    const jaga = data.hedgehogs.find(p => p.username === 'jaga');
        if (jaga) {
            html += `<div class="jaga-summary">
                <h4>🦔 Jaga's Status</h4>
                <p><strong>Qualities:</strong> Furriness: ${jaga.furriness}, Cuteness: ${jaga.cuteness}, Friendliness: ${jaga.friendliness}, Curiosity: ${jaga.curiosity}, Agility: ${jaga.agility}, Sleepiness: ${jaga.sleepiness}</p>
                <p><strong>AI Evaluation:</strong> <span class="evaluation-score ${data.jaga_evaluation >= 100 ? 'success' : 'normal'}">${data.jaga_evaluation ? data.jaga_evaluation.toFixed(2) : 'N/A'}</span></p>
            </div>`;
        }

    // Show hedgehogs table with evaluation scores
    html += '<div class="table-container"><table><thead><tr>';
    html += '<th>Hedgehog</th><th>Furriness</th><th>Cuteness</th><th>Friendliness</th><th>Curiosity</th><th>Agility</th><th>Sleepiness</th><th>AI Evaluation</th><th>Role</th>';
        html += '</tr></thead><tbody>';

    data.hedgehogs.forEach(hedgehog => {
        const isJaga = hedgehog.username === 'jaga';
        const evaluationScore = hedgehog.evaluation_score !== undefined ? hedgehog.evaluation_score.toFixed(2) : 'N/A';
        const storedScore = hedgehog.stored_evaluation_score !== undefined && hedgehog.stored_evaluation_score !== null ? hedgehog.stored_evaluation_score.toFixed(2) : null;
            const role = isJaga ? 'Test Data' : 'Training Data';
        const evaluationClass = isJaga && hedgehog.evaluation_score >= 100 ? 'success' : (isJaga ? 'target' : 'normal');
            
            // Check if evaluation score has changed from stored value
        const scoreChanged = !isJaga && storedScore && Math.abs(hedgehog.evaluation_score - hedgehog.stored_evaluation_score) > 0.01;
            
            html += `<tr ${isJaga ? 'class="target-hedgehog"' : ''}>`;
        html += `<td>${hedgehog.username}</td>`;
        html += `<td>${hedgehog.furriness}</td>`;
        html += `<td>${hedgehog.cuteness}</td>`;
        html += `<td>${hedgehog.friendliness}</td>`;
        html += `<td>${hedgehog.curiosity}</td>`;
        html += `<td>${hedgehog.agility}</td>`;
        html += `<td>${hedgehog.sleepiness}</td>`;
            html += `<td>`;
            html += `<span class="evaluation-score ${evaluationClass}">${evaluationScore}</span>`;
            if (scoreChanged) {
                html += `<br><small class="stored-score">Original: ${storedScore}</small>`;
            }
            html += `</td>`;
            html += `<td><span class="role-badge ${role.toLowerCase().replace(' ', '-')}">${role}</span></td>`;
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        
        // Add training data summary if available
        if (data.training_evaluations && data.training_evaluations.length > 0) {
            html += '<div class="training-summary">';
            html += '<h4>📊 Training Data Evaluation Summary</h4>';
            html += '<p>The AI model was trained on the following hedgehogs and their evaluation scores:</p>';
            html += '<ul>';
            data.training_evaluations.forEach(training => {
                const currentScore = training.evaluation_score.toFixed(2);
                const storedScore = training.stored_evaluation_score !== undefined && training.stored_evaluation_score !== null ? training.stored_evaluation_score.toFixed(2) : null;
                const scoreChanged = storedScore && Math.abs(training.evaluation_score - training.stored_evaluation_score) > 0.01;
                
                html += `<li><strong>${training.username}:</strong> ${currentScore}`;
                if (scoreChanged) {
                    html += ` <span class="stored-score">(was: ${storedScore})</span>`;
                }
                html += '</li>';
            });
            html += '</ul>';
            html += `<p><strong>Total training hedgehogs:</strong> ${data.training_count || data.training_evaluations.length}</p>`;
            if (data.training_evaluations.some(t => t.stored_evaluation_score !== undefined && Math.abs(t.evaluation_score - t.stored_evaluation_score) > 0.01)) {
                html += '<p class="data-warning">⚠️ <strong>Data changes detected!</strong> Some training data evaluation scores have changed from their original values.</p>';
            }
            html += '</div>';
        }

        resultsContent.innerHTML = html;
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Show notification
    function showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            max-width: 400px;
            animation: slideIn 0.3s ease-out;
        `;
        
        if (type === 'success') {
            notification.style.background = 'linear-gradient(135deg, #27ae60, #229954)';
        } else {
            notification.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';
        }
        
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 5000);
    }

    // Update jaga's evaluation display
    function updateJagaDisplay(score) {
    const jagaRows = document.querySelectorAll('.target-hedgehog');
        jagaRows.forEach(row => {
            // Add evaluation score indicator
            if (!row.querySelector('.evaluation-score')) {
                const scoreCell = document.createElement('td');
                scoreCell.className = 'evaluation-score';
                scoreCell.textContent = `Eval: ${score.toFixed(2)}`;
                scoreCell.style.cssText = `
                    background: ${score >= 100 ? '#2ecc71' : '#f39c12'};
                    color: white;
                    font-weight: bold;
                    text-align: center;
                `;
                row.appendChild(scoreCell);
            } else {
                const scoreCell = row.querySelector('.evaluation-score');
                scoreCell.textContent = `Eval: ${score.toFixed(2)}`;
                scoreCell.style.background = score >= 100 ? '#2ecc71' : '#f39c12';
            }
        });
    }

    // Show celebration animation
    function showCelebration() {
        // Create confetti effect
        for (let i = 0; i < 50; i++) {
            createConfetti();
        }
    }

    function createConfetti() {
        const confetti = document.createElement('div');
        confetti.style.cssText = `
            position: fixed;
            top: -10px;
            left: ${Math.random() * 100}vw;
            width: 10px;
            height: 10px;
            background: ${['#f1c40f', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][Math.floor(Math.random() * 5)]};
            pointer-events: none;
            z-index: 1001;
            animation: confetti 3s linear forwards;
        `;
        
        document.body.appendChild(confetti);
        
        setTimeout(() => {
            document.body.removeChild(confetti);
        }, 3000);
    }

    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        @keyframes confetti {
            0% { transform: translateY(0) rotate(0deg); opacity: 1; }
            100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Initialize application
    console.log('🦔 Best Hedgehog Evaluation System Loaded!');
});