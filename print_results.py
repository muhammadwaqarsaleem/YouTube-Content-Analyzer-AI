import glob, json

videos = {
    "Rollercoaster": "BQL-CP8d1-o",
    "WWII": "HUqy-OQvVtI",
    "Burglary": "1igfjbtkaL8",
    "Comedy": "rBCHIdqso3c"
}

with open("results_dump.txt", "w", encoding="utf-8") as out:
    for name, vid in videos.items():
        files = glob.glob(f"temp/results/analysis_{vid}_*.json")
        for f in files:
            data = json.load(open(f, 'r', encoding='utf-8'))
            if 'summary' in data:
                out.write(f"--- {name} ({vid}) ---\n")
                out.write(f"Violence: {json.dumps(data['summary'].get('violence', {}))}\n")
                out.write(f"Category: {json.dumps(data['summary'].get('category', {}))}\n\n")
