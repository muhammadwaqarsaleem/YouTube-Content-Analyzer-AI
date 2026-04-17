# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   Web Dashboard  │  │   REST API       │  │   Mobile App     │ │
│  │   (Frontend)     │  │   Clients        │  │   (Future)       │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY LAYER                           │
│                    FastAPI Application (api/main.py)                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Endpoints:                                                   │  │
│  │  • POST /analyze/video    - Single video analysis            │  │
│  │  • POST /analyze/batch    - Batch video analysis             │  │
│  │  • GET  /analysis/{id}    - Retrieve results                 │  │
│  │  • DELETE /analysis/{id}  - Delete analysis                  │  │
│  │  • GET  /health           - Health check                      │  │
│  │  • GET  /stats            - System statistics                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKGROUND TASK PROCESSOR                      │
│              Async Task Queue (FastAPI BackgroundTasks)             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Task Manager:                                                │  │
│  │  • Queue management                                           │  │
│  │  • Progress tracking                                          │  │
│  │  • Error handling                                             │  │
│  │  • Result caching                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       MEDIA EXTRACTION LAYER                        │
│              YouTubeMediaExtractor (src/youtube_extractor.py)       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   yt-dlp        │  │   FFmpeg        │  │   Transcript    │    │
│  │   Video Download│  │   Frame Extract │  │   API           │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│         ↓                      ↓                      ↓            │
│  temp/videos/          temp/frames/         temp/thumbnails/       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     PARALLEL PROCESSING LAYER                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  VIOLENCE DETECTION PIPELINE                                  │ │
│  │                                                               │ │
│  │  ViolenceFramePreprocessor                                    │ │
│  │    ↓                                                          │ │
│  │  Load & normalize frames (224x224 RGB)                       │ │
│  │    ↓                                                          │ │
│  │  ViolenceDetectionService                                     │ │
│  │    ↓                                                          │ │
│  │  ResNet50 Model (violence_detection_model_resnet.h5)         │ │
│  │    ↓                                                          │ │
│  │  Batch inference with memory management                       │ │
│  │    ↓                                                          │ │
│  │  Timeline generation with timestamps                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  CATEGORY PREDICTION PIPELINE                                 │ │
│  │                                                               │ │
│  │  CategoryFeatureExtractor                                     │ │
│  │    ↓                                                          │ │
│  │  Multi-modal feature extraction:                              │ │
│  │    • Image features (thumbnail pixels)                        │ │
│  │    • Text features (TF-IDF from title/desc/tags)              │ │
│  │    • Numerical features (views, duration, engagement)         │ │
│  │    • Categorical features (quality, platform)                 │ │
│  │    ↓                                                          │ │
│  │  CategoryPredictionService                                    │ │
│  │    ↓                                                          │ │
│  │  Logistic Regression Model                                    │ │
│  │  (models/logistic_regression_model.pkl)                       │ │
│  │    ↓                                                          │ │
│  │  Multi-label classification                                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      RESULTS AGGREGATION LAYER                      │
│               AnalysisAggregator (services/analysis_aggregator.py)  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Aggregation Steps:                                           │  │
│  │  1. Combine violence + category results                       │  │
│  │  2. Calculate summary metrics                                 │  │
│  │  3. Determine content rating (SAFE/CAUTION/RESTRICTED)       │  │
│  │  4. Generate recommendations                                  │  │
│  │  5. Format for dashboard display                              │  │
│  │  6. Save JSON report to temp/results/                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│                   Web Dashboard (frontend/*.html,*.js,*.css)        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Display Components:                                          │  │
│  │  • Video Information Card                                     │  │
│  │  • Content Rating Badge                                       │  │
│  │  • Violence Metrics Grid                                      │  │
│  │  • Interactive Violence Timeline                              │  │
│  │  • Category Prediction Bars                                   │  │
│  │  • Processing Information                                     │  │
│  │  • Download Report Button                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence

```
┌────────────┐
│   User     │
└─────┬──────┘
      │ 1. Paste YouTube URL
      ↓
┌─────────────────────────────────┐
│   Web Dashboard                 │
│   - Validate URL                │
│   - Send POST request           │
└─────┬───────────────────────────┘
      │ 2. POST /analyze/video
      ↓
┌─────────────────────────────────┐
│   FastAPI Server                │
│   - Generate analysis_id        │
│   - Start background task       │
│   - Return analysis_id          │
└─────┬───────────────────────────┘
      │ 3. Background processing starts
      ↓
┌─────────────────────────────────┐
│   YouTubeMediaExtractor         │
│   - Download video (yt-dlp)     │
│   - Extract ALL frames          │
│   - Extract thumbnail           │
│   - Get metadata                │
└─────┬───────────────────────────┘
      │ 4. Media extracted
      ↓
    ╱════════════════════════════════╲
   ╱     PARALLEL PROCESSING          ╲
  ╱                                    ╲
 ↓                                      ↓
┌──────────────────────┐    ┌──────────────────────┐
│ Violence Detection   │    │ Category Prediction  │
│                      │    │                      │
│ • Preprocess frames  │    │ • Extract features   │
│ • Batch prediction   │    │ • ML inference       │
│ • Generate timeline  │    │ • Calculate probs    │
└──────────┬───────────┘    └──────────┬───────────┘
           ╲                          ╱
            ╲                        ╱
             ╲                      ╱
              ↓                    ↓
        ┌─────────────────────────────────┐
        │   AnalysisAggregator            │
        │   - Combine results             │
        │   - Create summary              │
        │   - Format for display          │
        │   - Save JSON report            │
        └──────────┬──────────────────────┘
                   │ 5. Results ready
                   ↓
        ┌─────────────────────────────────┐
        │   Polling from Dashboard        │
        │   GET /analysis/{id}            │
        └──────────┬──────────────────────┘
                   │ 6. Return formatted data
                   ↓
        ┌─────────────────────────────────┐
        │   Dashboard Renders:            │
        │   - Violence timeline           │
        │   - Category bars               │
        │   - Content rating              │
        │   - Video info                  │
        └─────────────────────────────────┘
```

## Component Interaction Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        FILE STRUCTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  api/main.py ──────────────────► services/violence_service.py  │
│       │                              │                          │
│       │                              │ uses                     │
│       │                              ↓                          │
│       │                     predict_violence.py                │
│       │                              │                          │
│       │                              │ loads                    │
│       │                              ↓                          │
│       │                    violence_detection_model.h5         │
│       │                                                         │
│       │                              ┌──────────────────────┐  │
│       ├─────────────────────────────►│  services/           │  │
│       │                              │  category_service.py │  │
│       │                              │                      │  │
│       │                              │ uses                 │  │
│       │                              ↓                      │  │
│       │                     src/category_features.py        │  │
│       │                              │                      │  │
│       │                              │ loads                │  │
│       │                              ↓                      │  │
│       │                     features/*.pkl                  │  │
│       │                             models/*.pkl            │  │
│       │                                                         │
│       ├────────────────────────────────────────────────────┐   │
│       │                                                    │   │
│       ↓                                                    ↓   │
│  src/youtube_extractor.py                          utils/file_manager.py
│       │                                                    │
│       │ uses                                               │ manages
│       ↓                                                    ↓
│  yt-dlp                                            temp/
│  FFmpeg                                              ├── videos/
│                                                      ├── frames/
│                                                      ├── thumbnails/
│                                                      └── results/
│
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

```
┌──────────────────────────────────────────────────────────┐
│                    PRESENTATION                          │
│  HTML5 + CSS3 + JavaScript (Vanilla)                     │
│  Responsive Design + Animations                          │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                      API LAYER                           │
│  FastAPI (Python)                                        │
│  Uvicorn (ASGI Server)                                   │
│  Pydantic (Data Validation)                              │
│  CORS Middleware                                         │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC                         │
│  Python Services Layer                                   │
│  - ViolenceDetectionService                              │
│  - CategoryPredictionService                             │
│  - AnalysisAggregator                                    │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING                       │
│  TensorFlow 2.x (Keras)                                  │
│  ResNet50 (Violence Detection)                           │
│  Scikit-learn (Category Prediction)                      │
│  OpenCV (Image Processing)                               │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                  DATA EXTRACTION                         │
│  yt-dlp (YouTube Download)                               │
│  youtube-transcript-api (Subtitles)                      │
│  ffmpeg-python (Video Processing)                        │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                   UTILITIES                              │
│  NumPy (Numerical Operations)                            │
│  Pandas (Data Manipulation)                              │
│  Pickle/Joblib (Model Serialization)                     │
│  Pathlib (File Operations)                               │
└──────────────────────────────────────────────────────────┘
```

---

**This architecture provides:**
- ✅ Scalability (async processing, batch operations)
- ✅ Modularity (separate services, clear boundaries)
- ✅ Maintainability (well-documented, organized structure)
- ✅ Performance (GPU support, memory-efficient processing)
- ✅ User Experience (real-time updates, interactive dashboard)
