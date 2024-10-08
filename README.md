<h1 align="center">YOLO Object Detection and Tracking</h1>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.x-blue.svg" alt="Python 3.x">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-green.svg" alt="YOLO Object Detection">
  <img src="https://img.shields.io/badge/DeepSORT-Tracking-orange.svg" alt="DeepSORT Tracking">
</p>

<p align="center">  
    A real-time object detection and tracking system using YOLO 11 and Deep SORT.
</p>

---

<h2>📋 Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#dependencies">Dependencies</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#code-explanation">Code Explanation</a></li>
  <li><a href="#customization">Customization</a></li>
  <li><a href="#license">License</a></li>
</ul>

---

<h2 id="overview">📖 Overview</h2>
<p>
  This project implements real-time object detection and tracking using YOLO and Deep SORT. The tracking algorithm ensures persistent IDs for detected objects and handles detection across video frames.
</p>

---

<h2 id="features">🌟 Features</h2>

<ul>
  <li>Real-time object detection using YOLO.</li>
  <li>Deep SORT object tracking with ID persistence across frames.</li>
  <li>Customizable detection confidence threshold.</li>
  <li>Aspect ratio maintained using padding for resized images.</li>
  <li>Filter to track only objects that appear in the center of the frame.</li>
</ul>

---

<h2 id="dependencies">🛠️ Dependencies</h2>
<p>Make sure to install the following Python libraries:</p>

<pre><code>pip install opencv-python torch deep_sort_realtime numpy</code></pre>

<ul>
  <li><b>opencv-python</b> - For handling video frames and drawing bounding boxes.</li>
  <li><b>torch</b> - To load and run the YOLO model.</li>
  <li><b>deep_sort_realtime</b> - For object tracking across frames.</li>
  <li><b>numpy</b> - General-purpose array operations.</li>
</ul>

---

<h2 id="installation">💻 Installation</h2>

<ol>
  <li>Clone the repository:</li>

<pre><code>git clone https://github.com/iamrukeshduwal/yolov11_real_time_object_detection_with_DeepSORT.git
cd yolo-object-detection-tracking
</code></pre>

  <li>Install the required Python libraries:</li>

<pre><code>pip install -r requirements.txt</code></pre>

  <li>Ensure your YOLO model weights are placed in the correct directory and update the <code>MODEL_PATH</code> in <code>yolo_detection_tracker.py</code> accordingly.</li>
</ol>

---

<h2 id="usage">🚀 Usage</h2>
<p>Run the following command to start detecting and tracking objects in a video:</p>

<pre><code>python yolo_detection_tracker.py</code></pre>

<p>Modify the video path and parameters (e.g., confidence threshold) in <code>yolo_detection_tracker.py</code> to suit your needs.</p>

---

<h2 id="code-explanation">📝 Code Explanation</h2>

<h3><code>yolo_detection_tracker.py</code></h3>
<p>The main script that handles video input, object detection with YOLO, and tracking with Deep SORT.</p>

<pre><code>detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
tracker = Tracker()
</code></pre>

<p>Tracks objects, maintains their IDs, and only tracks objects in the middle of the frame.</p>

<h3><code>yolo_detector.py</code></h3>
<p>Contains the <b>YoloDetector</b> class that loads the YOLO model and performs object detection.</p>

<pre><code>detections = detector.detect(frame)
</code></pre>

<h3><code>tracker.py</code></h3>
<p>Defines the <b>Tracker</b> class, which implements object tracking using the Deep SORT algorithm.</p>

<pre><code>tracking_ids, boxes = tracker.track(detections, resized_frame)
</code></pre>

---

<h2 id="customization">⚙️ Customization</h2>

<h3>Adjusting Detection Confidence</h3>

<p>You can change the detection confidence threshold in the <code>YoloDetector</code> by modifying the following line in <code>yolo_detection_tracker.py</code>:</p>

<pre><code>detector = YoloDetector(model_path=MODEL_PATH, confidence=0.3)
</code></pre>

<h3>Filtering Objects by Position</h3>

<p>The current implementation only tracks objects detected in the middle of the frame. You can adjust this behavior in <code>yolo_detection_tracker.py</code> by modifying the center filtering logic.</p>

---

<h2 id="license">📜 License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more details.</p>

---

<h4 align="center">Developed by <a href="https://github.com/iamrukeshduwal">Rukesh Duwal</a> with 💖</h4>
