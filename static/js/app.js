// Visa Approval Prediction - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Initialize any interactive elements
    initializeAnimations();
});

function handleFormSubmit(event) {
    const form = event.target;
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    
    // Show loading state
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    btn.disabled = true;
    
    // Form will submit normally to Flask backend
    // If you want to use API instead, uncomment below:
    /*
    event.preventDefault();
    
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    data.PREVAILING_WAGE = parseFloat(data.PREVAILING_WAGE);
    data.YEAR = parseInt(data.YEAR);
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            showError(result.error);
            resetButton();
        } else {
            // Display result on same page
            displayResult(result);
            resetButton();
        }
    })
    .catch(error => {
        showError('An error occurred. Please try again.');
        resetButton();
    });
    */
}

function resetButton() {
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    
    btnText.style.display = 'inline';
    btnLoading.style.display = 'none';
    btn.disabled = false;
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
}

function displayResult(result) {
    // This function would update the page with results
    // For now, we're using server-side rendering
    console.log('Prediction result:', result);
}

function initializeAnimations() {
    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animate statistics on analytics page
    animateStats();
}

function animateStats() {
    const statValues = document.querySelectorAll('.stat-value');
    
    statValues.forEach(stat => {
        const finalValue = stat.textContent;
        const isNumeric = !isNaN(parseInt(finalValue.replace(/[^0-9]/g, '')));
        
        if (isNumeric) {
            const target = parseInt(finalValue.replace(/[^0-9]/g, ''));
            let current = 0;
            const increment = target / 50;
            const duration = 1000;
            const stepTime = duration / 50;
            
            const counter = setInterval(() => {
                current += increment;
                if (current >= target) {
                    stat.textContent = finalValue;
                    clearInterval(counter);
                } else {
                    stat.textContent = Math.floor(current).toLocaleString();
                }
            }, stepTime);
        }
    });
    
    // Animate progress bars
    const progressBars = document.querySelectorAll('.probability-fill, .bar');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

// Utility function to format currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Utility function to format percentage
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}
