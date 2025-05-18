/**
 * Feedback Validator Application
 * Main JavaScript file for handling all app functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const feedbackForm = {
        reasonSelect: document.getElementById('feedbackReason'),
        textArea: document.getElementById('feedbackText'),
        stars: document.querySelectorAll('.stars i'),
        ratingText: document.querySelector('.rating-text'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        clearFormBtn: document.getElementById('clearFormBtn')
    };
    
    const analysisStates = {
        emptyState: document.getElementById('emptyState'),
        loadingState: document.getElementById('loadingState'),
        resultsContainer: document.getElementById('resultsContainer')
    };
    
    const historyElements = {
        historyToggleBtn: document.getElementById('historyToggleBtn'),
        historyPanel: document.getElementById('historyPanel'),
        closeHistoryBtn: document.getElementById('closeHistoryBtn'),
        clearHistoryBtn: document.getElementById('clearHistoryBtn'),
        emptyHistory: document.getElementById('emptyHistory'),
        historyList: document.getElementById('historyList')
    };
    
    // State variables
    let currentRating = 0;
    let feedbackHistory = [];
    
    // Initialize the app
    initializeApp();
    
    /**
     * Initialize the application
     */
    function initializeApp() {
        // Setup event listeners
        setupRatingSystem();
        setupFormSubmission();
        setupHistoryPanel();
        
        // Load feedback history from the backend
        loadHistoryFromBackend();
    }
    
    /**
     * Setup the star rating system
     */
    function setupRatingSystem() {
        // Handle star hover effects
        feedbackForm.stars.forEach(star => {
            // Hover effects
            star.addEventListener('mouseover', () => {
                const rating = parseInt(star.getAttribute('data-rating'));
                updateStarDisplay(rating, 'hover');
            });
            
            // Click to set rating
            star.addEventListener('click', () => {
                currentRating = parseInt(star.getAttribute('data-rating'));
                updateStarDisplay(currentRating, 'selected');
            });
        });
        
        // Reset stars when mouse leaves the container
        document.querySelector('.stars').addEventListener('mouseleave', () => {
            updateStarDisplay(currentRating, 'selected');
        });
    }
    
    /**
     * Update the star display based on rating and state
     * @param {number} rating - The rating value (1-5)
     * @param {string} state - The state ('hover' or 'selected')
     */
    function updateStarDisplay(rating, state) {
        feedbackForm.stars.forEach(star => {
            const starRating = parseInt(star.getAttribute('data-rating'));
            
            // Reset all stars
            star.className = 'far fa-star';
            
            // Fill stars based on rating
            if (starRating <= rating) {
                star.className = 'fas fa-star';
            }
        });
        
        // Update rating text
        if (rating > 0) {
            feedbackForm.ratingText.textContent = `${rating} star${rating !== 1 ? 's' : ''}`;
        } else {
            feedbackForm.ratingText.textContent = 'No rating';
        }
    }
    
    /**
     * Setup form submission and analysis
     */
    function setupFormSubmission() {
        // Analyze button
        feedbackForm.analyzeBtn.addEventListener('click', () => {
            if (validateForm()) {
                analyzeFeedback();
            }
        });
        
        // Clear form button
        feedbackForm.clearFormBtn.addEventListener('click', clearForm);
    }
    
    /**
     * Validate the feedback form
     * @returns {boolean} - Whether the form is valid
     */
    function validateForm() {
        const reason = feedbackForm.reasonSelect.value;
        const text = feedbackForm.textArea.value.trim();
        
        // Basic validation
        if (!reason) {
            alert('Please select a feedback reason');
            return false;
        }
        
        if (text.length < 10) {
            alert('Please provide more detailed feedback (at least 10 characters)');
            return false;
        }
        
        if (currentRating === 0) {
            alert('Please provide a rating');
            return false;
        }
        
        return true;
    }
    
    /**
     * Analyze the feedback
     */
    function analyzeFeedback() {
        // Show loading state
        showAnalysisState('loading');
        
        // Prepare feedback data
        const feedbackData = {
            reason: feedbackForm.reasonSelect.value,
            text: feedbackForm.textArea.value.trim(),
            rating: currentRating
        };
        
        // Send to backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Update UI with results
            updateResultsDisplay(data.analysis);
            showAnalysisState('results');
            
            // Reload history
            loadHistoryFromBackend();
            
            // Update processing time
            document.getElementById('processingTime').querySelector('span').textContent = 
                `Processed successfully`;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error analyzing feedback. Please try again.');
            showAnalysisState('empty');
        });
    }
    
    /**
     * Load feedback history from backend
     */
    function loadHistoryFromBackend() {
        fetch('/history')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            feedbackHistory = data;
            updateHistoryDisplay();
        })
        .catch(error => {
            console.error('Error loading history:', error);
        });
    }
    
    /**
     * Update the results display with analysis data
     * @param {Object} analysis - The analysis results
     */
    function updateResultsDisplay(analysis) {
        // Update validation result
        const validationResult = document.getElementById('validationResult');
        const resultIcon = validationResult.querySelector('.result-icon');
        const resultTitle = validationResult.querySelector('.result-title');
        const resultConfidence = validationResult.querySelector('.result-confidence');
        
        resultIcon.className = `result-icon fas ${analysis.valid ? 'fa-check-circle' : 'fa-times-circle'}`;
        resultTitle.textContent = analysis.valid ? 'Valid Feedback' : 'Invalid Feedback';
        resultConfidence.textContent = `Confidence: ${analysis.confidence}%`;
        
        // Update sentiment
        document.getElementById('sentimentIcon').className = 
            `feature-icon fas ${analysis.sentiment.type === 'POSITIVE' ? 'fa-thumbs-up' : 
                             analysis.sentiment.type === 'NEGATIVE' ? 'fa-thumbs-down' : 'fa-meh'}`;
        document.getElementById('sentimentLabel').textContent = analysis.sentiment.type;
        document.getElementById('sentimentScore').textContent = `Score: ${analysis.sentiment.score}`;
        
        // Update grammar
        document.getElementById('grammarErrors').textContent = 
            `${analysis.grammar.errors} error${analysis.grammar.errors !== 1 ? 's' : ''}`;
        document.getElementById('readabilityScore').textContent = `Readability: ${analysis.grammar.readability}`;
        
        // Update specificity
        document.getElementById('specificityLevel').textContent = analysis.specificity.level;
        document.getElementById('specificityScore').textContent = `Score: ${analysis.specificity.score}/10`;
        
        // Update politeness
        document.getElementById('politenessLevel').textContent = analysis.politeness.level;
        document.getElementById('politenessScore').textContent = `Score: ${analysis.politeness.score}`;
        
        // Update temporal context
        document.getElementById('temporalStatus').textContent = analysis.temporal.status;
        document.getElementById('temporalExpressions').textContent = analysis.temporal.expressions;
        
        // Update relevance
        document.getElementById('relevanceLevel').textContent = analysis.relevance.level;
        document.getElementById('relevanceScore').textContent = `Score: ${analysis.relevance.score}/10`;
    }
    
    /**
     * Show an analysis state (empty, loading, or results)
     * @param {string} state - The state to show ('empty', 'loading', or 'results')
     */
    function showAnalysisState(state) {
        // Hide all states
        analysisStates.emptyState.classList.remove('active');
        analysisStates.loadingState.classList.remove('active');
        analysisStates.resultsContainer.classList.remove('active');
        
        // Show the requested state
        if (state === 'empty') {
            analysisStates.emptyState.classList.add('active');
        } else if (state === 'loading') {
            analysisStates.loadingState.classList.add('active');
        } else if (state === 'results') {
            analysisStates.resultsContainer.classList.add('active');
        }
    }
    
    /**
     * Setup history panel functionality
     */
    function setupHistoryPanel() {
        // Toggle history panel
        historyElements.historyToggleBtn.addEventListener('click', () => {
            historyElements.historyPanel.classList.toggle('active');
            
            // Update button text
            const buttonText = historyElements.historyToggleBtn.querySelector('span');
            buttonText.textContent = historyElements.historyPanel.classList.contains('active') ? 
                'Hide History' : 'Show History';
        });
        
        // Close history panel
        historyElements.closeHistoryBtn.addEventListener('click', () => {
            historyElements.historyPanel.classList.remove('active');
            
            // Update button text
            const buttonText = historyElements.historyToggleBtn.querySelector('span');
            buttonText.textContent = 'Show History';
        });
        
        // Clear history
        historyElements.clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all feedback history?')) {
                clearHistory();
            }
        });
    }
    
    /**
     * Update the history display
     */
    function updateHistoryDisplay() {
        // Show or hide empty state
        if (feedbackHistory.length === 0) {
            historyElements.emptyHistory.classList.add('active');
            historyElements.historyList.innerHTML = '';
            return;
        }
        
        historyElements.emptyHistory.classList.remove('active');
        
        // Update history list
        historyElements.historyList.innerHTML = '';
        
        feedbackHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            // Format date
            const date = new Date(item.timestamp);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            
            // Get feedback type display name
            const reasonDisplay = getFeedbackReasonDisplay(item.reason);
            
            // Create rating stars
            const stars = Array(5).fill(0).map((_, i) => 
                `<i class="${i < item.rating ? 'fas' : 'far'} fa-star small-star"></i>`
            ).join('');
            
            // Create HTML content
            historyItem.innerHTML = `
                <div class="history-item-header">
                    <div class="history-item-type">${reasonDisplay}</div>
                    <div class="history-item-date">${formattedDate}</div>
                </div>
                <div class="history-item-text">${truncateText(item.text, 100)}</div>
                <div class="history-item-footer">
                    <div class="history-item-rating">${stars}</div>
                    <div class="history-item-sentiment ${item.analysis.sentiment.type.toLowerCase()}">${item.analysis.sentiment.type}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon view-item" data-id="${item.id}">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn-icon remove-item" data-id="${item.id}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            
            // Add event listeners
            historyItem.querySelector('.view-item').addEventListener('click', () => {
                loadFeedbackItem(item.id);
            });
            
            historyItem.querySelector('.remove-item').addEventListener('click', () => {
                removeHistoryItem(item.id);
            });
            
            historyElements.historyList.appendChild(historyItem);
        });
    }
    
    /**
     * Get display name for feedback reason
     * @param {string} reason - The feedback reason code
     * @returns {string} - The display name
     */
    function getFeedbackReasonDisplay(reason) {
        const reasonMap = {
            'connection_issues': 'Connection Issues',
            'audio_quality': 'Audio Quality',
            'video_quality': 'Video Quality',
            'meeting_experience': 'Meeting Experience',
            'feature_request': 'Feature Request',
            'other': 'Other'
        };
        
        return reasonMap[reason] || reason;
    }
    
    /**
     * Load a feedback item into the form
     * @param {string} id - The ID of the feedback item
     */
    function loadFeedbackItem(id) {
        fetch(`/history/${id}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(item => {
            // Fill form
            feedbackForm.reasonSelect.value = item.reason;
            feedbackForm.textArea.value = item.text;
            currentRating = item.rating;
            updateStarDisplay(currentRating, 'selected');
            
            // Close history panel
            historyElements.historyPanel.classList.remove('active');
            historyElements.historyToggleBtn.querySelector('span').textContent = 'Show History';
            
            // Show analysis results
            updateResultsDisplay(item.analysis);
            showAnalysisState('results');
            
            // Update processing time
            document.getElementById('processingTime').querySelector('span').textContent = 'Loaded from history';
        })
        .catch(error => {
            console.error('Error loading feedback item:', error);
            alert('Error loading feedback item. Please try again.');
        });
    }
    
    /**
     * Remove a history item
     * @param {string} id - The ID of the item to remove
     */
    function removeHistoryItem(id) {
        if (confirm('Are you sure you want to remove this feedback item?')) {
            fetch(`/history/${id}`, {
                method: 'DELETE'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Reload history
                loadHistoryFromBackend();
            })
            .catch(error => {
                console.error('Error removing feedback item:', error);
                alert('Error removing feedback item. Please try again.');
            });
        }
    }
    
    /**
     * Clear the form
     */
    function clearForm() {
        feedbackForm.reasonSelect.value = '';
        feedbackForm.textArea.value = '';
        currentRating = 0;
        updateStarDisplay(0, 'selected');
        showAnalysisState('empty');
    }
    
    /**
     * Clear the history
     */
    function clearHistory() {
        fetch('/history', {
            method: 'DELETE'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            feedbackHistory = [];
            updateHistoryDisplay();
        })
        .catch(error => {
            console.error('Error clearing history:', error);
            alert('Error clearing history. Please try again.');
        });
    }
    
    /**
     * Truncate text to a maximum length
     * @param {string} text - The text to truncate
     * @param {number} maxLength - The maximum length
     * @returns {string} - The truncated text
     */
    function truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
});