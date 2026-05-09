# TextileGuard AI: Industrial Defect Detection

An advanced, real-time textile defect detection system using YOLOv8 and secondary classification. This system is designed to identify holes, stains, lines, and other anomalies in fabric production lines.

## 🚀 Features
- **Real-time Detection:** Powered by YOLOv8 for sub-50ms latency.
- **Secondary Validation:** Integrated classification model to reduce false positives using industrial datasets.
- **Multi-Class Support:** Detects holes, stains, needle marks, and structural anomalies.
- **Flexible Input:** Supports USB webcams and IP cameras (phone cameras via Wi-Fi/USB).
- **Customizable:** Simple configuration for production-line sensitivity and alerts.

## 🛠️ Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/textile-defect-detection.git
   cd textile-defect-detection
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Live Inspection:**
   ```bash
   python live_detection.py
   ```

## 📊 AI Architecture
- **Stage 1 (Detection):** YOLOv8 model trained on synthetic and real-world industrial datasets.
- **Stage 2 (Classification):** Secondary validation stage using the Kaggle Textile Patches dataset (70,000+ samples).
- **Filtering:** Custom temporal and spatial filters to ensure industrial-grade reliability.

## 📜 License
MIT License. See `LICENSE` for details.
