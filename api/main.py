"""
YouTube Video Analysis API
Main FastAPI application
"""

# Suppress TensorFlow/Keras log noise before any TF imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN custom ops messages

import warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')
warnings.filterwarnings('ignore', message='.*Valid config keys have changed.*')

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, HttpUrl, validator
from typing import List, Optional
from contextlib import asynccontextmanager
import time
from pathlib import Path
import sys
import yt_dlp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.violence_service import ViolenceDetectionService
from services.category_service import CategoryPredictionService
from services.analysis_aggregator import AnalysisAggregator
from src.youtube_extractor import YouTubeMediaExtractor
from utils.file_manager import FileManager
from utils.video_utils import validate_video_file, estimate_processing_time


# Global services (initialized once)
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("="*70)
    print("INITIALIZING YOUTUBE ANALYSIS API")
    print("="*70)
    
    try:
        # Initialize file manager
        services['file_manager'] = FileManager()
        
        # Initialize YouTube extractor
        services['extractor'] = YouTubeMediaExtractor()
        
        # Initialize violence detection service
        violence_model_path = 'models/violence_detection/violence_detection_model_resnet.h5'
        if Path(violence_model_path).exists():
            services['violence_service'] = ViolenceDetectionService(violence_model_path)
        else:
            print(f"Warning: Violence model not found at {violence_model_path}")
            services['violence_service'] = None
        
        # Initialize category prediction service (XGBoost model)
        category_model_path = 'models/xgboost_category_model.pkl'
        if not Path(category_model_path).exists():
            # Fallback to old logistic regression model
            category_model_path = 'models/logistic_regression_model.pkl'
        if Path(category_model_path).exists():
            services['category_service'] = CategoryPredictionService(category_model_path)
        else:
            print(f"Warning: Category model not found at {category_model_path}")
            services['category_service'] = None
        
        # Initialize aggregator
        services['aggregator'] = AnalysisAggregator()
        
        print("\n[OK] All services initialized successfully")
        
    except Exception as e:
        print(f"Error initializing services: {e}")
        raise
    
    print("="*70)
    
    yield  # Application runs here
    
    # Shutdown
    print("\nCleaning up temporary files...")
    if 'file_manager' in services:
        services['file_manager'].cleanup_old_files(max_age_hours=1)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="YouTube Video Analysis API",
    description="Analyze YouTube videos for violence detection and category prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class VideoAnalysisRequest(BaseModel):
    video_url: str
    frames_per_minute: Optional[int] = 10  # New parameter for dynamic framing
    
    @validator('frames_per_minute', pre=True, always=True)
    def validate_frames_per_minute(cls, v):
        """Validate frames per minute parameter"""
        if v is None:
            return 10  # Default value
        if v < 1:
            raise ValueError('frames_per_minute must be at least 1')
        if v > 60:
            raise ValueError('frames_per_minute cannot exceed 60')
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
            "frames_per_minute": 15
        }
    })


class BatchAnalysisRequest(BaseModel):
    video_urls: List[str]
    frames_per_minute: Optional[int] = 10


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    data: Optional[dict] = None
    error: Optional[str] = None

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "YouTube Video Analysis API",
        "version": "1.0.0",
        "description": "Analyze YouTube videos for violence and categorization",
        "endpoints": {
            "POST /analyze/video": "Analyze single video",
            "POST /analyze/batch": "Analyze multiple videos",
            "GET /analysis/{analysis_id}": "Get analysis results",
            "GET /health": "Health check",
            "GET /api/info": "This info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check each service
    for name, service in services.items():
        if service is not None:
            status["services"][name] = "available"
        else:
            status["services"][name] = "unavailable"
    
    return status


@app.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(
    request: VideoAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a single YouTube video
    
    Args:
        request: Analysis request with video URL
        background_tasks: FastAPI background tasks
        
    Returns:
        Analysis results
    """
    print(f"\nReceived analysis request for: {request.video_url}")
    
    start_time = time.time()
    
    try:
        # Extract video ID
        extractor = services['extractor']
        video_id = extractor.get_video_id(request.video_url)
        
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Generate analysis ID
        analysis_id = f"analysis_{video_id}_{int(time.time())}"
        
        # Process video in background
        background_tasks.add_task(
            process_video_async,
            video_url=request.video_url,
            video_id=video_id,
            analysis_id=analysis_id,
            frames_per_minute=request.frames_per_minute,
            start_time=start_time
        )
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message="Video analysis started. Use GET /analysis/{analysis_id} to retrieve results.",
            data={"estimated_time_seconds": estimate_processing_time_for_request(request)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Get analysis results by ID
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Analysis results or status
    """
    try:
        aggregator = services['aggregator']
        
        # Try to load results
        report = aggregator.load_report(analysis_id)
        
        if report:
            # Format for display
            display_data = aggregator.format_for_display(report)
            
            return {
                "analysis_id": analysis_id,
                "status": "complete",
                "data": display_data
            }
        else:
            # Check if processing
            result_path = Path('temp/results') / f"{analysis_id}.json"
            temp_path = Path('temp/results') / f"{analysis_id}_processing.json"
            error_path = Path('temp/results') / f"{analysis_id}_error.json"
            
            # Get file manager from services
            file_manager = services['file_manager']
            
            # Check for errors first
            if error_path.exists():
                error_data = file_manager.load_json(f"{analysis_id}_error.json")
                error_msg = error_data.get('error', 'Unknown error')
                error_type = error_data.get('error_type', 'Error')
                timestamp = error_data.get('timestamp', 0)
                
                # Calculate how long ago the error occurred
                time_ago = time.time() - timestamp if timestamp else 0
                
                print(f"❌ Analysis failed for {analysis_id}: {error_type} - {error_msg} ({time_ago:.1f}s ago)")
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {error_msg}",
                    headers={"X-Error-Type": error_type, "X-Error-Time": str(round(time_ago))}
                )
            
            if temp_path.exists():
                # Load processing status to get current step
                try:
                    processing_data = file_manager.load_json(f"{analysis_id}_processing.json")
                    step = processing_data.get('step', 'processing')
                    elapsed = time.time() - processing_data.get('started_at', time.time())
                    return {
                        "analysis_id": analysis_id,
                        "status": "processing",
                        "step": step,
                        "elapsed_seconds": round(elapsed, 1)
                    }
                except:
                    return {
                        "analysis_id": analysis_id,
                        "status": "processing"
                    }
            
            # Return 202 to indicate processing is accepted and ongoing
            return JSONResponse(
                status_code=202,
                content={"status": "pending", "detail": f"Analysis queued: {analysis_id}"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze multiple YouTube videos
    
    Args:
        request: Batch analysis request
        background_tasks: Background tasks
        
    Returns:
        Batch analysis status
    """
    print(f"\nReceived batch analysis request for {len(request.video_urls)} videos")
    
    analysis_ids = []
    
    for i, url in enumerate(request.video_urls):
        try:
            extractor = services['extractor']
            video_id = extractor.get_video_id(url)
            
            if not video_id:
                continue
            
            analysis_id = f"batch_{video_id}_{int(time.time())}_{i}"
            analysis_ids.append({
                'video_url': url,
                'analysis_id': analysis_id
            })
            
            # Add to background tasks
            background_tasks.add_task(
                process_video_async,
                video_url=url,
                video_id=video_id,
                analysis_id=analysis_id,
                frames_per_minute=request.frames_per_minute
            )
            
        except Exception as e:
            print(f"Error processing video {i}: {e}")
    
    return AnalysisResponse(
        analysis_id="batch_" + str(int(time.time())),
        status="processing",
        message=f"Started analysis for {len(analysis_ids)} videos",
        data={"analyses": analysis_ids}
    )


@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete analysis results
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Deletion status
    """
    try:
        file_manager = services['file_manager']
        
        # Extract video ID from analysis ID
        parts = analysis_id.split('_')
        if len(parts) >= 2:
            video_id = parts[1]
            file_manager.cleanup_video_files(video_id)
        
        # Delete report
        report_path = Path('temp/results') / f"{analysis_id}.json"
        if report_path.exists():
            report_path.unlink()
        
        return {"status": "deleted", "analysis_id": analysis_id}
        
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    file_manager = services['file_manager']
    stats = file_manager.get_storage_stats()
    
    return {
        "storage": stats,
        "active_services": sum(1 for s in services.values() if s is not None)
    }


# ─── FEEDBACK LOOP ───

from typing import Optional

class FeedbackRequest(BaseModel):
    analysis_id: str
    feedback_type: str  # "category" or "violence"
    is_correct: bool
    user_correction: Optional[str] = None  # e.g. the correct category name
    predicted_value: Optional[str] = None  # what the system predicted
    video_url: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Store user feedback on prediction accuracy.
    Feedback is appended to data/feedback.jsonl for future model retraining.
    """
    import json
    
    feedback_dir = Path("data")
    feedback_dir.mkdir(exist_ok=True)
    feedback_file = feedback_dir / "feedback.jsonl"
    
    entry = {
        "timestamp": time.time(),
        "analysis_id": feedback.analysis_id,
        "feedback_type": feedback.feedback_type,
        "is_correct": feedback.is_correct,
        "predicted_value": feedback.predicted_value,
        "user_correction": feedback.user_correction,
        "video_url": feedback.video_url
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    
    print(f"📝 Feedback recorded: {feedback.feedback_type} {'✅ correct' if feedback.is_correct else '❌ incorrect'} (analysis: {feedback.analysis_id})")
    
    return {"status": "recorded", "message": "Thank you for your feedback!"}


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get aggregated feedback statistics for monitoring prediction accuracy."""
    import json
    
    feedback_file = Path("data/feedback.jsonl")
    if not feedback_file.exists():
        return {"total": 0, "category": {"correct": 0, "incorrect": 0}, "violence": {"correct": 0, "incorrect": 0}}
    
    stats = {
        "category": {"correct": 0, "incorrect": 0, "corrections": {}},
        "violence": {"correct": 0, "incorrect": 0}
    }
    
    total = 0
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += 1
                ft = entry.get("feedback_type", "")
                
                if ft in stats:
                    if entry.get("is_correct"):
                        stats[ft]["correct"] += 1
                    else:
                        stats[ft]["incorrect"] += 1
                        
                        # Track category corrections for retraining
                        if ft == "category" and entry.get("user_correction"):
                            corr = entry["user_correction"]
                            stats["category"]["corrections"][corr] = stats["category"]["corrections"].get(corr, 0) + 1
            except json.JSONDecodeError:
                continue
    
    # Calculate accuracy
    for ft in ["category", "violence"]:
        t = stats[ft]["correct"] + stats[ft]["incorrect"]
        stats[ft]["accuracy"] = round(stats[ft]["correct"] / t * 100, 1) if t > 0 else None
        stats[ft]["total"] = t
    
    stats["total"] = total
    return stats


# Helper Functions
def process_video_async(video_url, video_id, analysis_id, frames_per_minute=10, 
                       start_time=None):
    """
    Process video asynchronously in background
    
    Args:
        video_url: YouTube video URL
        video_id: Video ID
        analysis_id: Analysis ID
        frames_per_minute: Frames to extract per minute (default: 10)
        start_time: Processing start time
    """
    # Initialize services BEFORE try block for error handling
    file_manager = services['file_manager']
    aggregator = services['aggregator']
    extractor = services['extractor']
    violence_service = services['violence_service']
    category_service = services['category_service']
    
    try:
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {video_id}")
        print(f"{'='*70}")
        
        # Save processing status
        temp_status = {
            'analysis_id': analysis_id,
            'status': 'processing',
            'started_at': time.time(),
            'step': 'initializing'
        }
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        
        # Step 1: Extract media and metadata
        print("\n[1/4] Extracting video and metadata...")
        temp_status['step'] = 'validating'
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        
        # PRE-VALIDATION: Quick URL format check only
        # Full video validation is handled by the main extractor which has
        # proper PO Token + cookie support for age-restricted content
        import re
        youtube_pattern = re.compile(
            r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
        )
        if not youtube_pattern.match(video_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid YouTube URL format."
            )
        print(f"🔍 URL format validated: {video_url}")
        
        temp_status['step'] = 'extracting'
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        
        extraction_result = extractor.process_video(
            video_url,
            frames_per_minute=frames_per_minute
        )
        
        # Step 2: Violence detection
        print("\n[2/4] Analyzing violence...")
        temp_status['step'] = 'violence_detection'
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        
        if violence_service and extraction_result['frame_paths']:
            # Extract frames based on frames_per_minute setting (no hard truncation)
            # Extractor now handles hybrid sampling (intro + body) up to max_frames
            frames_to_analyze = extraction_result['frame_paths']
            
            # Get transcript for context analysis (English-only)
            transcript = extractor.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            
            # First get category prediction for context
            category_result_temp = None
            if category_service:
                category_result_temp = category_service.predict_category(
                    extraction_result['thumbnail_path'],
                    extraction_result['metadata']
                )
            
            # Analyze with context-aware detection
            violence_result = violence_service.analyze_video_frames(
                frames_to_analyze,
                timestamps=list(range(len(frames_to_analyze))),
                category_prediction=category_result_temp,
                transcript=transcript,
                metadata=extraction_result['metadata']  # Pass metadata for content type detection
            )
        else:
            violence_result = {
                'is_violent': False,
                'violence_percentage': 0.0,
                'violent_frame_count': 0,
                'total_frames': 0,
                'timeline': []
            }
        
        # Step 3: Category prediction (skip if already done for violence context)
        print("\n[3/4] Predicting category...")
        temp_status['step'] = 'category_prediction'
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        
        # Use the category result from violence detection step if available
        if 'category_result_temp' in locals() and category_result_temp is not None:
            category_result = category_result_temp
            print("  ✓ Using category result from violence context analysis")
        elif category_service:
            category_result = category_service.predict_category(
                extraction_result['thumbnail_path'],
                extraction_result['metadata']
            )
        else:
            category_result = {
                'primary_category': 'Unknown',
                'primary_probability': 0.0,
                'all_categories': []
            }
        
        # Step 4: Aggregate results
        print("\n[4/4] Aggregating results...")
        temp_status['step'] = 'aggregating'
        file_manager.save_json(temp_status, f"{analysis_id}_processing.json")
        processing_time = time.time() - start_time if start_time else 0
        
        report = aggregator.aggregate_results(
            video_id=video_id,
            violence_result=violence_result,
            category_result=category_result,
            metadata=extraction_result['metadata'],
            processing_time=round(processing_time, 2)
        )
        
        # Save report
        output_path = aggregator.save_report(report, f"{analysis_id}.json")
        
        # Cleanup temporary files (optional - keep for now)
        # file_manager.cleanup_video_files(video_id)
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE: {analysis_id}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Categorize error for better user feedback
        error_msg = str(e)
        error_lower = error_msg.lower()
        
        if "video unavailable" in error_lower or "private" in error_lower or "deleted" in error_lower:
            user_friendly_error = "This video is private, deleted, or unavailable."
            error_code = "VIDEO_UNAVAILABLE"
        elif "region" in error_lower or "country" in error_lower:
            user_friendly_error = "This video is not available in your region."
            error_code = "REGION_RESTRICTED"
        elif "age" in error_lower or "restricted" in error_lower:
            user_friendly_error = "This video is age-restricted. Please sign in to continue."
            error_code = "AGE_RESTRICTED"
        elif "quota" in error_lower or "limit" in error_lower:
            user_friendly_error = "API quota exceeded. Please try again later."
            error_code = "QUOTA_EXCEEDED"
        elif "timeout" in error_lower or "network" in error_lower:
            user_friendly_error = "Network error. Please check your connection and try again."
            error_code = "NETWORK_ERROR"
        else:
            user_friendly_error = f"Processing error: {str(e)}"
            error_code = "PROCESSING_FAILED"
        
        # Save error status with timestamp and categorization
        error_status = {
            'analysis_id': analysis_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.time(),
            'error_type': type(e).__name__,
            'error_code': error_code,
            'user_friendly_message': user_friendly_error
        }
        file_manager.save_json(error_status, f"{analysis_id}_error.json")
        
        # Also save a simplified error report
        if 'aggregator' in locals():
            error_report = {
                'analysis_id': analysis_id,
                'status': 'error',
                'error': str(e),
                'error_code': error_code,
                'user_message': user_friendly_error,
                'timestamp': time.time()
            }
            file_manager.save_json(error_report, f"{analysis_id}_error.json")


def estimate_processing_time_for_request(request: VideoAnalysisRequest):
    """Estimate processing time based on request parameters"""
    # Rough estimates
    base_time = 10  # Base overhead
    frame_extraction_rate = 30  # FPS extraction speed
    
    # Assume average 180 seconds (3 min) video
    avg_video_duration_seconds = 180
    avg_video_frames = avg_video_duration_seconds * 30  # 5400 frames
    
    if request.frames_per_minute:
        # Calculate estimated frames based on fps setting
        estimated_frames = int(request.frames_per_minute * (avg_video_duration_seconds / 60))
        extraction_time = estimated_frames / frame_extraction_rate
    else:
        # Default: extract all frames
        estimated_frames = avg_video_frames
        extraction_time = estimated_frames / frame_extraction_rate
    
    processing_time = 30 if estimated_frames > 100 else 10  # Model inference
    
    return round(base_time + extraction_time + processing_time, 1)


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting YouTube Video Analysis API Server")
    print("="*70)
    print("\nServer will be available at:")
    print("  http://localhost:8000")
    print("  http://127.0.0.1:8000")
    print("\nAPI Documentation:")
    print("  http://localhost:8000/docs")
    print("  http://localhost:8000/redoc")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
