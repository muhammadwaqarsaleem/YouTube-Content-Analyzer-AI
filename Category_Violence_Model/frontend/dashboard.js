// YouTube Video Analysis Dashboard JavaScript

const API_BASE_URL = 'http://localhost:8000';
let currentAnalysisId = null;
let currentReportData = null;

/**
 * Analyze video from YouTube URL
 */
async function analyzeVideo() {
    const urlInput = document.getElementById('videoUrl');
    const framesPerMinuteElement = document.getElementById('framesPerMinute');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const videoUrl = urlInput.value.trim();
    
    if (!videoUrl) {
        alert('Please enter a YouTube URL');
        return;
    }
    
    // Validate YouTube URL
    if (!isValidYouTubeUrl(videoUrl)) {
        alert('Please enter a valid YouTube URL');
        return;
    }
    
    // Get and validate frames per minute
    let framesPerMinute = parseInt(framesPerMinuteElement.value) || 60;  // Default to 60 for maximum accuracy
    
    // Ensure within valid range
    if (framesPerMinute < 1) {
        framesPerMinute = 1;
        framesPerMinuteElement.value = 1;
    } else if (framesPerMinute > 60) {
        framesPerMinute = 60;
        framesPerMinuteElement.value = 60;
    }
    
    // Disable button and show loading
    analyzeBtn.disabled = true;
    showLoadingSection();
    
    try {
        // Send analysis request
        const response = await fetch(`${API_BASE_URL}/analyze/video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_url: videoUrl,
                frames_per_minute: framesPerMinute
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        currentAnalysisId = result.analysis_id;
        
        console.log('Analysis started:', currentAnalysisId, '- Frames per minute:', framesPerMinute);
        
        // Start polling for results
        pollForResults(currentAnalysisId);
        
    } catch (error) {
        console.error('Error starting analysis:', error);
        showError(error.message);
        analyzeBtn.disabled = false;
    }
}

/**
 * Poll for analysis results
 */
async function pollForResults(analysisId) {
    const maxAttempts = 1440; // 1 hour with 2.5s interval (for very long videos)
    let attempts = 0;
    
    // Wait a bit before starting to poll (give backend time to start processing)
    await sleep(2000); // Wait 2 seconds before first poll
    
    const pollInterval = setInterval(async () => {
        try {
            attempts++;
            
            const response = await fetch(`${API_BASE_URL}/analysis/${analysisId}`);
            
            if (response.ok) {
                const result = await response.json();
                
                console.log('Poll attempt', attempts, '- Status:', result.status);
                
                if (result.status === 'complete') {
                    clearInterval(pollInterval);
                    updateLoadingStatus(attempts, maxAttempts, 'Analysis complete!');
                    displayResults(result.data);
                    currentReportData = result.data;
                } else if (result.status === 'processing') {
                    // Show detailed step info if available
                    if (result.step) {
                        const stepMessages = {
                            'initializing': '🔄 Initializing analysis...',
                            'extracting': '📥 Downloading and extracting video frames...',
                            'violence_detection': '🔍 Analyzing frames for violence...',
                            'category_prediction': '🧠 Predicting video category...',
                            'aggregating': '📊 Generating final report...'
                        };
                        const message = stepMessages[result.step] || '⏳ Processing video...';
                        updateLoadingStatus(attempts, maxAttempts, message);
                    } else {
                        updateLoadingStatus(attempts, maxAttempts, '⏳ Processing video...');
                    }
                    
                    // Show elapsed time if available
                    if (result.elapsed_seconds) {
                        const elapsedMins = Math.floor(result.elapsed_seconds / 60);
                        const elapsedSecs = Math.floor(result.elapsed_seconds % 60);
                        console.log(`Processing for ${elapsedMins}m ${elapsedSecs}s...`);
                    }
                } else {
                    throw new Error('Analysis failed');
                }
            } else if (response.status === 404) {
                // Still processing, no status file yet
                console.log('Poll attempt', attempts, '- Analysis not found yet, still processing...');
                updateLoadingStatus(attempts, maxAttempts, 'Starting analysis...');
            } else if (response.status === 500) {
                // Analysis failed - STOP POLLING immediately
                clearInterval(pollInterval);
                const errorData = await response.json().catch(() => ({}));
                const errorMessage = errorData.detail || 'Analysis failed due to server error';
                console.error('❌ Analysis failed:', errorMessage);
                showError(errorMessage);
                return; // Stop polling
            }
            
            if (attempts >= maxAttempts) {
                clearInterval(pollInterval);
                throw new Error('Analysis timed out after 1 hour. Please try again with a shorter video or check backend logs.');
            }
            
        } catch (error) {
            console.error('Polling error:', error);
            clearInterval(pollInterval);
            showError(error.message);
        }
    }, 2500); // Poll every 2.5 seconds
}

/**
 * Update loading status
 */
function updateLoadingStatus(attempt, maxAttempts, customMessage) {
    const progressFill = document.getElementById('progressFill');
    const statusText = document.getElementById('statusText');
    
    // Calculate progress based on typical processing stages, not timeout
    // Most videos complete in 5-15 minutes, so we'll base progress on that
    const typicalProcessingTime = 10 * 60; // 10 minutes in seconds
    const elapsedSeconds = attempt * 2.5;
    
    // Progress caps at 95% until completion
    let progress;
    if (elapsedSeconds < typicalProcessingTime) {
        // Normal progress based on typical time
        progress = Math.min((elapsedSeconds / typicalProcessingTime) * 90, 90);
    } else {
        // After typical time, slowly creep up to 95%
        progress = Math.min(90 + ((elapsedSeconds - typicalProcessingTime) / typicalProcessingTime) * 5, 95);
    }
    
    progressFill.style.width = `${progress}%`;
    
    // Update step indicators based on current message
    updateProgressSteps(customMessage);
    
    if (customMessage) {
        statusText.textContent = customMessage;
        
        // Add elapsed time info
        const elapsedMinutes = Math.floor(elapsedSeconds / 60);
        const elapsedSecondsDisplay = Math.floor(elapsedSeconds % 60);
        const timeInfo = document.getElementById('loadingText');
        if (timeInfo) {
            timeInfo.textContent = `Elapsed time: ${elapsedMinutes}m ${elapsedSecondsDisplay}s`;
        }
    } else {
        const messages = [
            'Downloading video...',
            'Extracting frames...',
            'Analyzing violence...',
            'Predicting category...',
            'Generating report...'
        ];
        
        const messageIndex = Math.floor((attempt / (typicalProcessingTime / 2.5)) * messages.length);
        statusText.textContent = messages[Math.min(messageIndex, messages.length - 1)] || 'Processing...';
        
        // Add elapsed time info
        const elapsedMinutes = Math.floor(elapsedSeconds / 60);
        const elapsedSecondsDisplay = Math.floor(elapsedSeconds % 60);
        const timeInfo = document.getElementById('loadingText');
        if (timeInfo) {
            timeInfo.textContent = `Elapsed time: ${elapsedMinutes}m ${elapsedSecondsDisplay}s`;
        }
    }
}

/**
 * Update progress step indicators
 */
function updateProgressSteps(message) {
    // Reset all steps first
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.classList.remove('active', 'completed');
        }
    }
    
    // Determine current step based on message
    let currentStep = 1;
    if (message.includes('Initializing')) currentStep = 1;
    else if (message.includes('Downloading') || message.includes('Extracting')) currentStep = 2;
    else if (message.includes('Analyzing')) currentStep = 3;
    else if (message.includes('Predicting')) currentStep = 4;
    else if (message.includes('Generating') || message.includes('complete')) currentStep = 5;
    
    // Mark completed steps
    for (let i = 1; i < currentStep; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.classList.add('completed');
        }
    }
    
    // Mark current step as active
    const currentStepEl = document.getElementById(`step${currentStep}`);
    if (currentStepEl) {
        currentStepEl.classList.add('active');
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    hideLoadingSection();
    showResultsSection();
    
    // Video Information
    document.getElementById('videoTitle').textContent = data.videoInfo.title || 'N/A';
    document.getElementById('videoChannel').textContent = data.videoInfo.channel || 'N/A';
    document.getElementById('videoDuration').textContent = data.videoInfo.duration || 'N/A';
    document.getElementById('videoViews').textContent = data.videoInfo.views || '0';
    
    // Overall Rating
    const ratingBadge = document.getElementById('overallRating');
    const rating = data.rating.overall;
    ratingBadge.textContent = rating;
    ratingBadge.className = 'rating-badge ' + rating;
    
    // Use the aggregator's unified recommendation (single source of truth)
    const recommendationEl = document.getElementById('contentRecommendation');
    recommendationEl.textContent = data.rating.recommendation || '';
    
    // Show cross-signal modifier if applied
    if (data.crossSignal && data.crossSignal.modifier !== 0) {
        const crossNote = document.createElement('small');
        crossNote.style.display = 'block';
        crossNote.style.marginTop = '8px';
        crossNote.style.color = '#6b7280';
        crossNote.style.fontStyle = 'italic';
        crossNote.textContent = `ℹ ${data.crossSignal.reason}`;
        recommendationEl.parentNode.appendChild(crossNote);
    }
    
    // Violence Assessment - SIMPLIFIED
    const violencePct = data.violenceMetrics.percentage;
    const severity = data.violenceMetrics.severity || 'NONE';
    const maxConfidence = data.violenceMetrics.max_confidence || 0;
    const recommendation = data.violenceMetrics.recommendation || '';
    const isViolent = data.violenceMetrics.isViolent || false;
    
    // Main violence indicator (BIG, CLEAR)
    const violenceIndicator = document.getElementById('violencePercentage');
    if (isViolent) {
        violenceIndicator.textContent = `${violencePct.toFixed(0)}% VIOLENCE DETECTED`;
        violenceIndicator.style.color = severity === 'EXTREME' || severity === 'HIGH' ? '#ef4444' : '#f59e0b';
        violenceIndicator.style.fontSize = '2.5em';
        violenceIndicator.style.fontWeight = 'bold';
    } else {
        violenceIndicator.textContent = 'NO VIOLENCE DETECTED';
        violenceIndicator.style.color = '#10b981';
        violenceIndicator.style.fontSize = '2em';
    }
    
    // Simple severity badge
    const severityBadge = document.getElementById('severityLevel');
    severityBadge.textContent = severity;
    severityBadge.className = `severity-badge ${severity.toLowerCase()}`;
    
    // Display max confidence in details section
    const maxConfEl = document.getElementById('maxConfidence');
    if (maxConfEl) {
        maxConfEl.textContent = isViolent ? `${(maxConfidence * 100).toFixed(1)}%` : 'N/A';
    }
    
    // Show assessment label in details section
    const assessmentEl = document.getElementById('violenceAssessment');
    if (assessmentEl) {
        assessmentEl.textContent = isViolent ? severity : 'No violence detected';
    }
    
    // REMOVE timeline rendering - complements the model, doesn't burdenize
    // renderViolenceTimeline(data.violenceMetrics.timeline);  // COMMENTED OUT
    
    // Category Prediction
    const primaryCategory = data.categoryMetrics.primary || 'Unknown';
    const categoryConf = data.categoryMetrics.confidence || 0;
    
    // Confidence is now pre-normalized to 0-1 by the aggregator
    const displayConfidence = (categoryConf * 100).toFixed(1);
    
    document.getElementById('primaryCategory').textContent = primaryCategory;
    document.getElementById('categoryConfidence').textContent = 
        `${displayConfidence}%`;
    
    // Add confidence indicator badge
    const confidenceIndicator = getConfidenceIndicator(categoryConf);
    const confidenceBadge = document.getElementById('confidenceBadge');
    if (confidenceBadge) {
        confidenceBadge.innerHTML = `${confidenceIndicator.emoji} ${confidenceIndicator.text}`;
        confidenceBadge.className = `confidence-badge ${confidenceIndicator.class}`;
        confidenceBadge.title = confidenceIndicator.description;
        confidenceBadge.style.display = 'inline-block';
    }
    
    // Display override decision explanation
    if (data.categoryMetrics.overrideDetails) {
        displayOverrideDetails(data.categoryMetrics.overrideDetails);
    }
    
    // Render category bars
    renderCategoryBars(data.categoryMetrics.categories);
    
    // Multi-label tag
    const multiLabelTag = document.getElementById('multiLabelTag');
    if (data.categoryMetrics.isMultiLabel) {
        multiLabelTag.style.display = 'inline-block';
    } else {
        multiLabelTag.style.display = 'none';
    }
    
    // Processing Info
    document.getElementById('processingTime').textContent = 
        `${data.processingInfo.time?.toFixed(1) || 0}s`;
    document.getElementById('analysisId').textContent = currentAnalysisId;
    
    // Populate feedback section
    const feedbackViolence = document.getElementById('feedbackViolenceValue');
    if (feedbackViolence) {
        feedbackViolence.textContent = isViolent ? `${severity} (${violencePct.toFixed(0)}%)` : 'None';
    }
    const feedbackCategory = document.getElementById('feedbackCategoryValue');
    if (feedbackCategory) {
        feedbackCategory.textContent = primaryCategory;
    }
    
    // Reset feedback buttons (in case user analyzes another video)
    ['violence', 'category'].forEach(type => {
        const btns = document.getElementById(`${type}FeedbackBtns`);
        const thanks = document.getElementById(`${type}FeedbackThanks`);
        if (btns) btns.style.display = 'flex';
        if (thanks) thanks.style.display = 'none';
    });
    const corrBox = document.getElementById('categoryCorrectionBox');
    if (corrBox) corrBox.style.display = 'none';
}

/**
 * Render violence timeline
 */
function renderViolenceTimeline(timeline) {
    const timelineContainer = document.getElementById('violenceTimeline');
    timelineContainer.innerHTML = '';
    
    if (!timeline || timeline.length === 0) {
        timelineContainer.innerHTML = '<p style="color: var(--text-secondary);">No frame data available</p>';
        return;
    }
    
    // Sample timeline if too many frames (show max 200)
    const sampleRate = Math.ceil(timeline.length / 200);
    
    timeline.forEach((frame, index) => {
        if (index % sampleRate === 0) {
            const frameDiv = document.createElement('div');
            frameDiv.className = `timeline-frame ${frame.is_violent ? 'violent' : 'safe'}`;
            frameDiv.title = `Frame ${index}: ${frame.is_violent ? 'Violent' : 'Safe'} (${(frame.probability_violence * 100).toFixed(0)}%)`;
            
            frameDiv.addEventListener('click', () => {
                alert(`Frame ${index}\nTimestamp: ${(frame.timestamp || 0).toFixed(2)}s\n${frame.label}\nConfidence: ${(frame.confidence * 100).toFixed(1)}%`);
            });
            
            timelineContainer.appendChild(frameDiv);
        }
    });
}

/**
 * Toggle override details visibility
 */
function toggleOverrideDetails() {
    const content = document.getElementById('overrideDetailsContent');
    const btn = document.getElementById('toggleOverrideBtn');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        btn.innerHTML = '🔽 Hide Why This Category';
    } else {
        content.style.display = 'none';
        btn.innerHTML = '🔍 Why This Category? <span id="overrideSummary"></span>';
    }
}

/**
 * Display override decision explanation
 */
function displayOverrideDetails(overrideDetails) {
    const container = document.getElementById('overrideDetailsContent');
    const toggleBtn = document.getElementById('toggleOverrideBtn');
    const summarySpan = document.getElementById('overrideSummary');
    
    if (!overrideDetails) {
        toggleBtn.style.display = 'none';
        return;
    }
    
    toggleBtn.style.display = 'inline-block';
    
    if (overrideDetails.triggered) {
        // Override WAS triggered - show why
        summarySpan.textContent = `(${overrideDetails.detector} detected at ${overrideDetails.score}/${overrideDetails.max_score})`;
        
        const indicatorsList = overrideDetails.indicators.map(ind => `<li>${ind}</li>`).join('');
        
        container.innerHTML = `
            <div class="override-explanation" style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 15px; border-radius: 8px;">
                <h4 style="color: #059669; margin-bottom: 10px;">✅ Override Applied: ${overrideDetails.detector}</h4>
                <p style="margin: 8px 0; color: #374151;"><strong>Original Model Prediction:</strong> ${overrideDetails.original_model_prediction} (${(overrideDetails.original_model_confidence * 100).toFixed(1)}%)</p>
                <p style="margin: 8px 0; color: #374151;"><strong>Override Score:</strong> ${overrideDetails.score}/${overrideDetails.max_score} (threshold: 4/10)</p>
                <p style="margin: 8px 0; color: #374151;"><strong>Detected Signals:</strong></p>
                <ul style="margin: 8px 0 0 20px; color: #374151;">
                    ${indicatorsList}
                </ul>
                <p style="margin: 12px 0 0 0; color: #059669; font-weight: 600;">→ Category forced to "${overrideDetails.detector}" based on strong metadata signals</p>
            </div>
        `;
    } else {
        // No override triggered - show model was used
        summarySpan.textContent = "(Model prediction)";
        
        container.innerHTML = `
            <div class="override-explanation" style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 8px;">
                <h4 style="color: #2563eb; margin-bottom: 10px;">ℹ️ Model Prediction Used (No Override)</h4>
                <p style="margin: 8px 0; color: #374151;"><strong>Predicted Category:</strong> ${overrideDetails.model_prediction_used} (${(overrideDetails.model_confidence * 100).toFixed(1)}%)</p>
                <p style="margin: 8px 0; color: #374151;"><strong>Reason:</strong> ${overrideDetails.reason}</p>
                <p style="margin: 12px 0 0 0; color: #6b7280; font-size: 0.9em;">No category-specific detector found strong enough signals (>4/10) to override the ML model's prediction.</p>
            </div>
        `;
    }
}

/**
 * Get confidence indicator badge based on confidence score
 */
function getConfidenceIndicator(confidence) {
    const confidencePct = confidence * 100;
    if (confidencePct >= 90) {
        return { 
            emoji: '🟢', 
            text: 'Very Confident', 
            class: 'confidence-high',
            description: 'This prediction is highly reliable'
        };
    } else if (confidencePct >= 70) {
        return { 
            emoji: '🟡', 
            text: 'Confident', 
            class: 'confidence-medium',
            description: 'This prediction is moderately reliable'
        };
    } else {
        return { 
            emoji: '🔴', 
            text: 'Uncertain', 
            class: 'confidence-low',
            description: 'Manual review recommended'
        };
    }
}

/**
 * Render category prediction bars
 */
function renderCategoryBars(categories) {
    const container = document.getElementById('categoryBars');
    container.innerHTML = '';
    
    if (!categories || categories.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary);">No category data available</p>';
        return;
    }
    
    // Show top 5 categories
    categories.slice(0, 5).forEach(cat => {
        const barItem = document.createElement('div');
        barItem.className = 'category-bar-item';
        
        const label = document.createElement('div');
        label.className = 'category-bar-label';
        label.textContent = cat.category;
        
        const track = document.createElement('div');
        track.className = 'category-bar-track';
        
        const fill = document.createElement('div');
        fill.className = 'category-bar-fill';
        fill.style.width = '0%';
        
        const value = document.createElement('div');
        value.className = 'category-bar-value';
        value.textContent = `${(cat.probability * 100).toFixed(1)}%`;
        
        track.appendChild(fill);
        barItem.appendChild(label);
        barItem.appendChild(track);
        barItem.appendChild(value);
        container.appendChild(barItem);
        
        // Animate bar after delay
        setTimeout(() => {
            fill.style.width = `${cat.probability * 100}%`;
        }, 100);
    });
}

/**
 * Download analysis report
 */
function downloadReport() {
    if (!currentReportData) {
        alert('No report data available');
        return;
    }
    
    const report = {
        analysis_id: currentAnalysisId,
        timestamp: new Date().toISOString(),
        data: currentReportData
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${currentAnalysisId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Analyze another video
 */
function analyzeAnother() {
    document.getElementById('videoUrl').value = '';
    currentAnalysisId = null;
    currentReportData = null;
    
    hideResultsSection();
    showInputSection();
    document.getElementById('analyzeBtn').disabled = false;
}

/**
 * Reset to input section
 */
function resetToInput() {
    hideErrorSection();
    resetToInput();
}

/**
 * Show/hide sections
 */
function showLoadingSection() {
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
}

function hideLoadingSection() {
    document.getElementById('loadingSection').style.display = 'none';
}

function showResultsSection() {
    document.getElementById('resultsSection').style.display = 'grid';
    document.getElementById('resultsSection').classList.add('fade-in');
}

function hideResultsSection() {
    document.getElementById('resultsSection').style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
}

function hideErrorSection() {
    document.getElementById('errorSection').style.display = 'none';
}

function showInputSection() {
    document.querySelector('.input-section').style.display = 'block';
}

/**
 * Validate YouTube URL
 */
function isValidYouTubeUrl(url) {
    const patterns = [
        /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/i,
        /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/i,
        /^https?:\/\/youtu\.be\/[\w-]+/i
    ];
    
    return patterns.some(pattern => pattern.test(url));
}

/**
 * Initialize dashboard
 */
document.addEventListener('DOMContentLoaded', () => {
    // Add Enter key support for URL input
    document.getElementById('videoUrl').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            analyzeVideo();
        }
    });
    
    // Slider event listener for real-time value update
    const framesSlider = document.getElementById('framesPerMinute');
    const fpsValueDisplay = document.getElementById('framesPerMinuteValue');
    const estimatedFramesDisplay = document.getElementById('estimatedFrames');
    
    if (framesSlider) {
        framesSlider.addEventListener('input', function(e) {
            const fps = e.target.value;
            fpsValueDisplay.textContent = fps;
            
            // Calculate estimated frames (assume avg 3 min video = 180 seconds)
            const estimatedFrames = Math.round(fps * 3); // 3 minutes default estimate
            estimatedFramesDisplay.textContent = `~${estimatedFrames}`;
        });
        
        // Initialize with default value
        const defaultValue = framesSlider.value;
        fpsValueDisplay.textContent = defaultValue;
        estimatedFramesDisplay.textContent = `~${Math.round(defaultValue * 3)}`;
    }
    
    console.log('YouTube Video Analysis Dashboard initialized');
    console.log('API Base URL:', API_BASE_URL);
});

/**
 * Sleep helper function
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── FEEDBACK LOOP HANDLERS ───
