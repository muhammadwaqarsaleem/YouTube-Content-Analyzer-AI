"""
Flask web server for the YouTube Content Analyzer dashboard.
Models are loaded once at startup and reused across requests.
"""
import os
import sys
import logging

# ── Suppress TF noise before heavy imports ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 stdout on Windows
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

# ── Load orchestrator once ──────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    logger.error('YOUTUBE_API_KEY not set. Please add it to your .env file.')
    sys.exit(1)

logger.info('Loading GlobalOrchestrator and all ML models...')
from global_analyzer import GlobalOrchestrator
orchestrator = GlobalOrchestrator(api_key=YOUTUBE_API_KEY)
logger.info('All models loaded. Server is ready.')


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data or not data.get('url', '').strip():
        return jsonify({'error': 'Please provide a valid YouTube URL.'}), 400

    url = data['url'].strip()
    logger.info(f'Web analysis request for: {url}')

    try:
        result = orchestrator.get_results_as_dict(url)
        return jsonify(result)
    except ValueError as e:
        logger.warning(f'Invalid URL or video not found: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f'Unexpected error during analysis: {e}')
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
