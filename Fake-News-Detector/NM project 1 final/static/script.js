document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const predictForm = document.getElementById('predict-form');
    const resultContainer = document.getElementById('result-container');
    const resultTitle = document.getElementById('result-title');
    const resultMessage = document.getElementById('result-message');
    const realProbability = document.getElementById('real-probability');
    const fakeProbability = document.getElementById('fake-probability');
    const realFill = document.getElementById('real-fill');
    const fakeFill = document.getElementById('fake-fill');
    
    // Add event listener to form submission
    if (predictForm) {
        predictForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            // Get text input
            const textInput = document.getElementById('text-input').value;
            
            if (!textInput.trim()) {
                alert('Please enter some text to analyze');
                return;
            }
            
            // Show loading state
            resultContainer.style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            
            // Make API request
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textInput })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Update result
                if (data.prediction === 1) {
                    resultContainer.className = 'result-card result-real';
                    resultTitle.textContent = 'Real News';
                    resultMessage.textContent = 'This post is classified as real news with high confidence.';
                } else {
                    resultContainer.className = 'result-card result-fake';
                    resultTitle.textContent = 'Fake News';
                    resultMessage.textContent = 'This post is classified as fake news with high confidence.';
                }
                
                // Update probabilities
                const realProb = Math.round(data.real_probability * 100);
                const fakeProb = Math.round(data.fake_probability * 100);
                
                realProbability.textContent = `${realProb}%`;
                fakeProbability.textContent = `${fakeProb}%`;
                
                realFill.style.width = `${realProb}%`;
                fakeFill.style.width = `${fakeProb}%`;
                
                // Show result
                resultContainer.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                alert('An error occurred while processing your request. Please try again.');
            });
        });
    }
    
    // Initialize visualizations in dashboard
    const visualizationSelect = document.getElementById('visualization-select');
    const visualizationImage = document.getElementById('visualization-image');
    
    if (visualizationSelect && visualizationImage) {
        visualizationSelect.addEventListener('change', function() {
            const selectedVisualization = this.value;
            if (selectedVisualization) {
                visualizationImage.src = `/static/${selectedVisualization}`;
                visualizationImage.style.display = 'block';
            } else {
                visualizationImage.style.display = 'none';
            }
        });
    }
});
