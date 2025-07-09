import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from tracking import run_monitoring

# Configuration
BASE_VIDEO_FOLDER = 'videos'
INPUT_FOLDER = os.path.join(BASE_VIDEO_FOLDER, 'input')
OUTPUT_FOLDER = os.path.join(BASE_VIDEO_FOLDER, 'output')
FEEDBACK_FOLDER = 'feedback'
TRACKING_JSON = os.path.join(OUTPUT_FOLDER, "tracking.json")
FRAME_FOLDER = os.path.join(BASE_VIDEO_FOLDER, "frames")

INPUT_VIDEO = os.path.join(INPUT_FOLDER, "input.mp4")
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, "monitoring_output.mp4")

# Create folders
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("file")
    if uploaded_file:
        uploaded_file.save(INPUT_VIDEO)
        run_monitoring(INPUT_VIDEO, OUTPUT_VIDEO)
        return redirect(url_for("processing_done"))
    return redirect("/")


@app.route("/done")
def processing_done():
    video_filename = os.path.basename(OUTPUT_VIDEO)
    return render_template("result.html", output_video=url_for('serve_output_video', filename=video_filename))


@app.route("/videos/output/<filename>")
def serve_output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype='video/mp4')


@app.route("/submit_correction", methods=["POST"])
def submit_correction():
    try:
        frame_number = int(request.form.get("frame_id"))
        object_id = request.form.get("track_id")  # Keep as string to match JSON
        object_type = request.form.get("correct_type")
        label = request.form.get("correct_label")

        if not all([frame_number >= 0, object_id, object_type, label]):
            return jsonify({"error": "Missing or invalid parameters"}), 400

        if not os.path.exists(TRACKING_JSON):
            return jsonify({"error": "Tracking data not found"}), 404

        with open(TRACKING_JSON) as f:
            tracking_data = json.load(f)

        # Find the target object in JSON
        target_obj = next(
            (obj for obj in tracking_data
             if obj["frame"] == frame_number and str(obj["id"]) == str(object_id)),
            None
        )

        if not target_obj:
            return jsonify({"error": "Object not found in specified frame"}), 404

        # Load the original frame
        frame_path = os.path.join(FRAME_FOLDER, f"frame_{frame_number}.jpg")
        if not os.path.exists(frame_path):
            return jsonify({"error": "Frame image not found"}), 404

        frame = cv2.imread(frame_path)
        if frame is None:
            return jsonify({"error": "Failed to read frame image"}), 500

        # Crop and save corrected object
        x1, y1, x2, y2 = map(int, target_obj["bbox"])
        crop = frame[y1:y2, x1:x2]

        save_dir = os.path.join(FEEDBACK_FOLDER, object_type, label)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"frame{frame_number}_id{object_id}.jpg")
        cv2.imwrite(save_path, crop)

        return jsonify({
            "message": "Correction saved successfully",
            "path": save_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
