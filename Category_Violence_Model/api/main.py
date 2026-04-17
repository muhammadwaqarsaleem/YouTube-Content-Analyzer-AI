"""
YouTube Video Analysis API
Main FastAPI application - Refactored for clean code practices
"""

import os
import re
import sys
import json
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

# Suppress TensorFlow/Keras log noise before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')
warnings.filterwarnings('ignore', message='.*Valid config keys have changed.*')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.violence_service import ViolenceDetectionService
from services.category_service import CategoryPredictionService
from services.analysis_aggregator import AnalysisAggregator
from src.youtube_extractor import YouTubeMediaExtractor
from utils.file_manager import FileManager
from utils.video_utils import estimate_processing_time


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class AppConfig:
    """Application configuration constants"""
    TITLE = "YouTube Video Analysis API"
    VERSION = "1.0.0"
    DESCRIPTION = "Analyze YouTube videos for violence detection and category prediction"
    CORS_ORIGIN = "http://localhost:3000"
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Model paths
    VIOLENCE_MODEL_PATH = "models/violence_detection/fine_tuned/final_model.h5"  # Updated to fine-tuned model
    CATEGORY_MODEL_PRIMARY = "models/xgboost_category_model.pkl"
    CATEGORY_MODEL_FALLBACK = "models/logistic_regression_model.pkl"
    
    # Processing defaults
    DEFAULT_FRAMES_PER_MINUTE = 60  # Maximum accuracy (changed from 10)
    MIN_FRAMES_PER_MINUTE = 1
    MAX_FRAMES_PER_MINUTE = 60


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class VideoAnalysisRequest(BaseModel):
    """Request model for single video analysis"""
    video_url: str
    frames_per_minute: Optional[int] = AppConfig.DEFAULT_FRAMES_PER_MINUTE
    
    @field_validator('frames_per_minute', mode='before')
    @classmethod
    def validate_frames_per_minute(cls, v: Optional[int]) -> int:
        """Validate frames per minute parameter"""
        if v is None:
            return AppConfig.DEFAULT_FRAMES_PER_MINUTE
        if v < AppConfig.MIN_FRAMES_PER_MINUTE:
            raise ValueError(f'frames_per_minute must be at least {AppConfig.MIN_FRAMES_PER_MINUTE}')
        if v > AppConfig.MAX_FRAMES_PER_MINUTE:
            raise ValueError(f'frames_per_minute cannot exceed {AppConfig.MAX_FRAMES_PER_MINUTE}')
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
                "frames_per_minute": 15
            }
        }
    )


class BatchAnalysisRequest(BaseModel):
    """Request model for batch video analysis"""
    video_urls: List[str]
    frames_per_minute: Optional[int] = AppConfig.DEFAULT_FRAMES_PER_MINUTE


class AnalysisResponse(BaseModel):
    """Response model for analysis operations"""
    analysis_id: str
    status: str
    message: str
    data: Optional[dict] = None
    error: Optional[str] = None


# ============================================================================
# SERVICE MANAGER
# ============================================================================

class ServiceManager:
    """Manages initialization and lifecycle of application services"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
    
    @property
    def file_manager(self) -> Optional[FileManager]:
        return self._services.get('file_manager')
    
    @property
    def extractor(self) -> Optional[YouTubeMediaExtractor]:
        return self._services.get('extractor')
    
    @property
    def violence_service(self) -> Optional[ViolenceDetectionService]:
        return self._services.get('violence_service')
    
    @property
    def category_service(self) -> Optional[CategoryPredictionService]:
        return self._services.get('category_service')
    
    @property
    def aggregator(self) -> Optional[AnalysisAggregator]:
        return self._services.get('aggregator')
    
    def initialize(self) -> None:
        """Initialize all application services"""
        print("="*70)
        print("INITIALIZING YOUTUBE ANALYSIS API")
        print("="*70)
        
        try:
            self._initialize_file_manager()
            self._initialize_extractor()
            self._initialize_violence_service()
            self._initialize_category_service()
            self._initialize_aggregator()
            
            print("\n[OK] All services initialized successfully")
        except Exception as e:
            print(f"Error initializing services: {e}")
            raise
        
        print("="*70)
    
    def _initialize_file_manager(self) -> None:
        """Initialize file manager service"""
        self._services['file_manager'] = FileManager()
        print("✓ File manager initialized")
    
    def _initialize_extractor(self) -> None:
        """Initialize YouTube media extractor"""
        self._services['extractor'] = YouTubeMediaExtractor()
        print("✓ YouTube extractor initialized")
    
    def _initialize_violence_service(self) -> None:
        """Initialize violence detection service"""
        model_path = AppConfig.VIOLENCE_MODEL_PATH
        if Path(model_path).exists():
            self._services['violence_service'] = ViolenceDetectionService(model_path)
            print(f"✓ Violence detection service initialized ({model_path})")
        else:
            print(f"⚠️  Violence model not found at {model_path}")
            self._services['violence_service'] = None
    
    def _initialize_category_service(self) -> None:
        """Initialize category prediction service with fallback"""
        model_path = AppConfig.CATEGORY_MODEL_PRIMARY
        
        if not Path(model_path).exists():
            print(f"⚠️  Primary category model not found, using fallback")
            model_path = AppConfig.CATEGORY_MODEL_FALLBACK
        
        if Path(model_path).exists():
            self._services['category_service'] = CategoryPredictionService(model_path)
            print(f"✓ Category service initialized ({model_path})")
        else:
            print(f"⚠️  Category model not found at {model_path}")
            self._services['category_service'] = None
    
    def _initialize_aggregator(self) -> None:
        """Initialize analysis aggregator"""
        self._services['aggregator'] = AnalysisAggregator()
        print("✓ Analysis aggregator initialized")
    
    def cleanup(self) -> None:
        """Cleanup services on shutdown"""
        print("\nCleaning up temporary files...")
        if self.file_manager:
            self.file_manager.cleanup_old_files(max_age_hours=1)
    
    def get_available_services(self) -> Dict[str, str]:
        """Get status of all services"""
        status = {}
        for name, service in self._services.items():
            status[name] = "available" if service is not None else "unavailable"
        return status


# Global service manager instance
services = ServiceManager()


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    services.initialize()
    yield
    # Shutdown
    services.cleanup()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=AppConfig.TITLE,
    description=AppConfig.DESCRIPTION,
    version=AppConfig.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[AppConfig.CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS - INFORMATION
# ============================================================================

@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": AppConfig.TITLE,
        "version": AppConfig.VERSION,
        "description": "Analyze YouTube videos for violence and categorization",
        "endpoints": {
            "POST /analyze/video": "Analyze single video",
            "POST /analyze/batch": "Analyze multiple videos",
            "GET /analysis/{analysis_id}": "Get analysis results",
            "GET /health": "Health check",
            "GET /stats": "Storage statistics",
            "GET /api/info": "This info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - returns service availability status"""
    return {
        "status": "healthy",
        "services": services.get_available_services()
    }


@app.get("/stats")
async def get_stats():
    """Get API storage statistics"""
    if not services.file_manager:
        return {"error": "File manager not available"}
    
    stats = services.file_manager.get_storage_stats()
    return {
        "storage": stats,
        "active_services": sum(1 for s in services.get_available_services().values() if s == "available")
    }


# ============================================================================
# API ENDPOINTS - VIDEO ANALYSIS
# ============================================================================

@app.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(
    request: VideoAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a single YouTube video
    
    Args:
        request: Analysis request containing video URL and optional frames_per_minute
        background_tasks: FastAPI background task manager
        
    Returns:
        AnalysisResponse with analysis ID and status
    """
    print(f"\nReceived analysis request for: {request.video_url}")
    
    start_time = time.time()
    
    try:
        # Validate and extract video ID
        if not services.extractor:
            raise HTTPException(status_code=503, detail="Extractor service unavailable")
        
        video_id = services.extractor.get_video_id(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Generate unique analysis ID
        analysis_id = f"analysis_{video_id}_{int(time.time())}"
        
        # Queue background processing - OPTIMAL 30 FPS
        background_tasks.add_task(
            _process_video_background,
            video_url=request.video_url,
            video_id=video_id,
            analysis_id=analysis_id,
            frames_per_minute=30,  # SWEET SPOT: Balanced accuracy and false positive reduction
            start_time=start_time
        )
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message="Video analysis started. Use GET /analysis/{analysis_id} to retrieve results.",
            data={"estimated_time_seconds": _estimate_processing_time(request)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Retrieve analysis results by ID
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        Analysis results or current processing status
    """
    if not services.aggregator or not services.file_manager:
        raise HTTPException(status_code=503, detail="Services unavailable")
    
    # Try to load completed report
    report = services.aggregator.load_report(analysis_id)
    
    if report:
        display_data = services.aggregator.format_for_display(report)
        return {
            "analysis_id": analysis_id,
            "status": "complete",
            "data": display_data
        }
    
    # Check processing status
    return _get_processing_status(analysis_id)


def _get_processing_status(analysis_id: str) -> dict:
    """Get current processing status for an analysis"""
    file_manager = services.file_manager
    
    error_path = Path('temp/results') / f"{analysis_id}_error.json"
    temp_path = Path('temp/results') / f"{analysis_id}_processing.json"
    
    # Check for errors first
    if error_path.exists():
        return _handle_processing_error(error_path, analysis_id, file_manager)
    
    # Check if still processing
    if temp_path.exists():
        return _get_current_processing_status(temp_path, analysis_id)
    
    # Still queued/pending
    return JSONResponse(
        status_code=202,
        content={"status": "pending", "detail": f"Analysis queued: {analysis_id}"}
    )


def _handle_processing_error(error_path: Path, analysis_id: str, file_manager: FileManager) -> None:
    """Handle processing error and raise appropriate HTTP exception"""
    error_data = file_manager.load_json(f"{analysis_id}_error.json")
    error_msg = error_data.get('error', 'Unknown error')
    error_type = error_data.get('error_type', 'Error')
    timestamp = error_data.get('timestamp', 0)
    
    time_ago = time.time() - timestamp if timestamp else 0
    print(f"❌ Analysis failed for {analysis_id}: {error_type} - {error_msg} ({time_ago:.1f}s ago)")
    
    raise HTTPException(
        status_code=500,
        detail=f"Analysis failed: {error_msg}",
        headers={
            "X-Error-Type": error_type,
            "X-Error-Time": str(round(time_ago))
        }
    )


def _get_current_processing_status(temp_path: Path, analysis_id: str) -> dict:
    """Get current processing step and elapsed time"""
    try:
        processing_data = services.file_manager.load_json(f"{analysis_id}_processing.json")
        step = processing_data.get('step', 'processing')
        elapsed = time.time() - processing_data.get('started_at', time.time())
        
        return {
            "analysis_id": analysis_id,
            "status": "processing",
            "step": step,
            "elapsed_seconds": round(elapsed, 1)
        }
    except Exception:
        return {
            "analysis_id": analysis_id,
            "status": "processing"
        }


@app.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze multiple YouTube videos in batch
    
    Args:
        request: Batch analysis request with list of video URLs
        background_tasks: FastAPI background task manager
        
    Returns:
        AnalysisResponse with batch status and individual analysis IDs
    """
    print(f"\nReceived batch analysis request for {len(request.video_urls)} videos")
    
    if not services.extractor:
        raise HTTPException(status_code=503, detail="Extractor service unavailable")
    
    analyses = []
    
    for i, url in enumerate(request.video_urls):
        try:
            video_id = services.extractor.get_video_id(url)
            if not video_id:
                continue
            
            analysis_id = f"batch_{video_id}_{int(time.time())}_{i}"
            analyses.append({
                'video_url': url,
                'analysis_id': analysis_id
            })
            
            background_tasks.add_task(
                _process_video_background,
                video_url=url,
                video_id=video_id,
                analysis_id=analysis_id,
                frames_per_minute=request.frames_per_minute
            )
            
        except Exception as e:
            print(f"Error processing video {i}: {e}")
    
    return AnalysisResponse(
        analysis_id=f"batch_{int(time.time())}",
        status="processing",
        message=f"Started analysis for {len(analyses)} videos",
        data={"analyses": analyses}
    )


@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete analysis results and associated files
    
    Args:
        analysis_id: Analysis ID to delete
        
    Returns:
        Deletion confirmation
    """
    if not services.file_manager:
        raise HTTPException(status_code=503, detail="File manager unavailable")
    
    try:
        # Extract video ID and cleanup associated files
        parts = analysis_id.split('_')
        if len(parts) >= 2:
            video_id = parts[1]
            services.file_manager.cleanup_video_files(video_id)
        
        # Delete report file
        report_path = Path('temp/results') / f"{analysis_id}.json"
        if report_path.exists():
            report_path.unlink()
        
        return {"status": "deleted", "analysis_id": analysis_id}
        
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def _process_video_background(
    video_url: str,
    video_id: str,
    analysis_id: str,
    frames_per_minute: int = 60,  # FIXED at 60 FPS - maximum accuracy always
    start_time: float = None
):
    """
    Process video asynchronously in background task
    
    This function handles the complete analysis pipeline:
    1. Extract media and metadata
    2. Detect violence
    3. Predict category
    4. Aggregate and save results
    """
    # Get service instances
    file_manager = services.file_manager
    aggregator = services.aggregator
    extractor = services.extractor
    violence_service = services.violence_service
    category_service = services.category_service
    
    if not all([file_manager, aggregator, extractor]):
        print("❌ Required services not available for background processing")
        return
    
    try:
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {video_id}")
        print(f"{'='*70}")
        
        # Initialize processing status
        _update_processing_status(file_manager, analysis_id, 'initializing', start_time)
        
        # Step 1: Extract media and metadata
        print("\n[1/4] Extracting video and metadata...")
        _update_processing_status(file_manager, analysis_id, 'extracting', start_time)
        
        _validate_youtube_url_format(video_url)
        extraction_result = extractor.process_video(video_url, frames_per_minute=frames_per_minute)
        
        # Step 2: Violence detection
        print("\n[2/4] Analyzing violence...")
        _update_processing_status(file_manager, analysis_id, 'violence_detection', start_time)
        
        violence_result = _analyze_violence(
            violence_service, category_service, extractor,
            extraction_result, video_id
        )
        
        # Step 3: Category prediction
        print("\n[3/4] Predicting category...")
        _update_processing_status(file_manager, analysis_id, 'category_prediction', start_time)
        
        category_result = _predict_category(
            category_service, extraction_result, violence_result
        )
        
        # Step 4: Aggregate results
        print("\n[4/4] Aggregating results...")
        _update_processing_status(file_manager, analysis_id, 'aggregating', start_time)
        
        processing_time = time.time() - start_time if start_time else 0
        
        report = aggregator.aggregate_results(
            video_id=video_id,
            violence_result=violence_result,
            category_result=category_result,
            metadata=extraction_result['metadata'],
            processing_time=round(processing_time, 2)
        )
        
        # Save final report
        output_path = aggregator.save_report(report, f"{analysis_id}.json")
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE: {analysis_id}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        _handle_processing_error_background(
            file_manager, analysis_id, e, traceback.format_exc()
        )


def _update_processing_status(
    file_manager: FileManager,
    analysis_id: str,
    step: str,
    start_time: float = None
):
    """Update processing status JSON file"""
    temp_status = {
        'analysis_id': analysis_id,
        'status': 'processing',
        'started_at': start_time or time.time(),
        'step': step
    }
    file_manager.save_json(temp_status, f"{analysis_id}_processing.json")


def _validate_youtube_url_format(video_url: str) -> None:
    """Validate YouTube URL format"""
    youtube_pattern = re.compile(
        r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+'
    )
    if not youtube_pattern.match(video_url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL format."
        )
    print(f"🔍 URL format validated: {video_url}")


def _analyze_violence(
    violence_service: Optional[ViolenceDetectionService],
    category_service: Optional[CategoryPredictionService],
    extractor: YouTubeMediaExtractor,
    extraction_result: dict,
    video_id: str
) -> dict:
    """Perform violence analysis on extracted frames"""
    
    if not violence_service or not extraction_result.get('frame_paths'):
        return {
            'is_violent': False,
            'violence_percentage': 0.0,
            'violent_frame_count': 0,
            'total_frames': 0,
            'timeline': []
        }
    
    frames_to_analyze = extraction_result['frame_paths']
    transcript = extractor.get_transcript(
        video_id, languages=['en', 'en-US', 'en-GB']
    )
    
    # Get category prediction for context (if available)
    category_result_temp = None
    if category_service:
        category_result_temp = category_service.predict_category(
            extraction_result['thumbnail_path'],
            extraction_result['metadata']
        )
    
    return violence_service.analyze_video_frames(
        frames_to_analyze,
        timestamps=list(range(len(frames_to_analyze))),
        category_prediction=category_result_temp,
        transcript=transcript,
        metadata=extraction_result['metadata']
    )


def _predict_category(
    category_service: Optional[CategoryPredictionService],
    extraction_result: dict,
    violence_result: dict
) -> dict:
    """Predict video category from thumbnail and metadata"""
    
    # Reuse category result from violence context if available
    if violence_result.get('category_result_temp') is not None:
        print("  ✓ Using category result from violence context analysis")
        return violence_result['category_result_temp']
    
    if not category_service:
        return {
            'primary_category': 'Unknown',
            'primary_probability': 0.0,
            'all_categories': []
        }
    
    return category_service.predict_category(
        extraction_result['thumbnail_path'],
        extraction_result['metadata']
    )


def _handle_processing_error_background(
    file_manager: FileManager,
    analysis_id: str,
    exception: Exception,
    traceback_str: str
):
    """Handle errors during background processing and save error report"""
    
    error_msg = str(exception)
    error_lower = error_msg.lower()
    
    # Categorize error for user-friendly messages
    error_mapping = {
        ("video unavailable", "private", "deleted"): 
            ("VIDEO_UNAVAILABLE", "This video is private, deleted, or unavailable."),
        ("region", "country"): 
            ("REGION_RESTRICTED", "This video is not available in your region."),
        ("age", "restricted"): 
            ("AGE_RESTRICTED", "This video is age-restricted. Please sign in to continue."),
        ("quota", "limit"): 
            ("QUOTA_EXCEEDED", "API quota exceeded. Please try again later."),
        ("timeout", "network"): 
            ("NETWORK_ERROR", "Network error. Please check your connection and try again.")
    }
    
    error_code = "PROCESSING_FAILED"
    user_friendly_message = f"Processing error: {error_msg}"
    
    for keywords, (code, message) in error_mapping.items():
        if any(keyword in error_lower for keyword in keywords):
            error_code = code
            user_friendly_message = message
            break
    
    # Save detailed error status
    error_status = {
        'analysis_id': analysis_id,
        'status': 'error',
        'error': error_msg,
        'traceback': traceback_str,
        'timestamp': time.time(),
        'error_type': type(exception).__name__,
        'error_code': error_code,
        'user_friendly_message': user_friendly_message
    }
    file_manager.save_json(error_status, f"{analysis_id}_error.json")
    
    # Save simplified error report
    error_report = {
        'analysis_id': analysis_id,
        'status': 'error',
        'error': error_msg,
        'error_code': error_code,
        'user_message': user_friendly_message,
        'timestamp': time.time()
    }
    file_manager.save_json(error_report, f"{analysis_id}_error.json")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _estimate_processing_time(request: VideoAnalysisRequest) -> float:
    """
    Estimate processing time based on request parameters
    
    Args:
        request: Video analysis request
        
    Returns:
        Estimated processing time in seconds
    """
    # Configuration constants
    BASE_TIME = 10  # Base overhead in seconds
    FRAME_EXTRACTION_RATE = 30  # Frames per second extraction speed
    AVG_VIDEO_DURATION_SECONDS = 180  # Assume 3-minute average video
    
    # Calculate estimated frames
    if request.frames_per_minute:
        estimated_frames = int(
            request.frames_per_minute * (AVG_VIDEO_DURATION_SECONDS / 60)
        )
    else:
        estimated_frames = AVG_VIDEO_DURATION_SECONDS * 30  # Default: all frames
    
    # Calculate processing components
    extraction_time = estimated_frames / FRAME_EXTRACTION_RATE
    inference_time = 30 if estimated_frames > 100 else 10
    
    return round(BASE_TIME + extraction_time + inference_time, 1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting YouTube Video Analysis API Server")
    print("="*70)
    print(f"\nServer will be available at:")
    print(f"  http://{AppConfig.HOST}:{AppConfig.PORT}")
    print(f"  http://127.0.0.1:{AppConfig.PORT}")
    print(f"\nAPI Documentation:")
    print(f"  http://localhost:{AppConfig.PORT}/docs")
    print(f"  http://localhost:{AppConfig.PORT}/redoc")
    print("="*70 + "\n")
    
    uvicorn.run(app, host=AppConfig.HOST, port=AppConfig.PORT)
