"""
Configuration constants for the YouTube Impact Score Model (v9).
"""

IMPACT_THRESHOLDS = {
    'minimal': (0,  20),
    'low':     (20, 40),
    'moderate':(40, 60),
    'high':    (60, 80),
    'viral':   (80, 100),
}

# Feature weights (v9)
# quality    0.30 — transcript & content quality now primary signal
# engagement 0.25 — most measurable but not dominant
# sentiment  0.20 — reduced; shallow comments are unreliable
# reach      0.15 — de-emphasised so low-reach quality content isn't buried
# virality   0.10 — 
FEATURE_WEIGHTS = {
    'quality':    0.30,
    'engagement': 0.25,
    'sentiment':  0.20,
    'reach':      0.15,
    'virality':   0.10,
}
