"""
Flask web server for the YouTube Content Analyzer dashboard.
Models are loaded once at startup and reused across requests.
Analysis results are cached in SQLite so repeat requests return instantly.
"""
import os
import re
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, render_template

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    logger.error('YOUTUBE_API_KEY not set. Please add it to your .env file.')
    sys.exit(1)

logger.info('Loading GlobalOrchestrator and all ML models...')
from global_analyzer import GlobalOrchestrator
orchestrator = GlobalOrchestrator(api_key=YOUTUBE_API_KEY)
logger.info('All models loaded. Server is ready.')

from analysis_cache import AnalysisCache
cache = AnalysisCache()
logger.info('Analysis cache initialised.')


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL string."""
    patterns = [
        r'(?:youtube\.com/watch\?v=)([^&]+)',
        r'(?:youtu\.be/)([^?]+)',
        r'(?:youtube\.com/embed/)([^?]+)',
        r'(?:youtube\.com/v/)([^?]+)',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return url


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data or not data.get('url', '').strip():
        return jsonify({'error': 'Please provide a valid YouTube URL.'}), 400

    url = data['url'].strip()
    force = data.get('force', False)

    video_id = _extract_video_id(url)
    logger.info(f'Web analysis request for: {url} (video_id={video_id}, force={force})')

    if not force:
        cached = cache.get(video_id)
        if cached:
            logger.info(f'Returning cached result for {video_id}')
            return jsonify(cached)

    try:
        result = orchestrator.get_results_as_dict(url)

        # Store in cache
        cache.put(video_id, url, result)

        return jsonify(result)
    except ValueError as e:
        logger.warning(f'Invalid URL or video not found: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f'Unexpected error during analysis: {e}')
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/history', methods=['GET'])
def history():
    """Return a list of previously analyzed videos."""
    try:
        entries = cache.list_history(limit=50)
        return jsonify(entries)
    except Exception as e:
        logger.exception(f'Error fetching history: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/history/<video_id>', methods=['DELETE'])
def delete_history(video_id):
    """Delete a single cached analysis entry."""
    try:
        deleted = cache.delete(video_id)
        if deleted:
            return jsonify({'status': 'deleted'})
        return jsonify({'error': 'Entry not found'}), 404
    except Exception as e:
        logger.exception(f'Error deleting history entry: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/history/<video_id>/result', methods=['GET'])
def get_cached_result(video_id):
    """Return the full cached result for a specific video."""
    try:
        result = cache.get(video_id)
        if result:
            return jsonify(result)
        return jsonify({'error': 'No cached result found'}), 404
    except Exception as e:
        logger.exception(f'Error fetching cached result: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
