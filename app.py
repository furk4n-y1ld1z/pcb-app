from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import io
import requests

# 1. Page Configuration
st.set_page_config(page_title="PCB Defect Inspector", page_icon="🔍", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
# SAHI will seamlessly wrap the Ultralytics engine to use your ONNX (or .pt) model
MODEL_PATH = BASE_DIR / "best.onnx" 

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Could not find model file at {MODEL_PATH}")
        return None
    # Initialize SAHI's YOLO wrapper
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(MODEL_PATH),
        device="cpu"
    )

model = load_model()

# 2. Sidebar Settings
st.sidebar.title("⚙️ Inspector Controls")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05
)

if model:
    st.sidebar.caption(f"Recognized Classes: {len(model.category_mapping)}")
    with st.sidebar.expander("View Defect Classes"):
        for idx, name in model.category_mapping.items():
            st.write(f"- `{idx}`: {name}")

# 3. Main Dashboard Header
st.title("PCB Defect Inspector 🔍")

# 4. Input Selection (Upload vs Camera)
st.markdown("### Select Input Source")
input_method = st.radio("Choose how to inspect the board:", ("📁 File Upload", "📷 Live Camera"), horizontal=True)

uploaded_file = None
if input_method == "📁 File Upload":
    uploaded_file = st.file_uploader("Drag and drop a PCB image (JPG/PNG)", type=["jpg", "jpeg", "png"])
elif input_method == "📷 Live Camera":
    st.info("Position your PCB under the camera and click 'Take Photo'.")
    uploaded_file = st.camera_input("Live Inspection Camera")

if uploaded_file is not None and model is not None:
    # Convert uploaded image to numpy array for SAHI and OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Dynamically apply the slider threshold
    model.confidence_threshold = confidence_threshold

    with st.spinner("Slicing high-res image and analyzing tiles ..."):
        # 5. Run Sliced Inference
        result = get_sliced_prediction(
            image_np,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # 6. Parse and Draw Results
        annotated_rgb = image_np.copy()
        defect_data = []
        
        for obj in result.object_prediction_list:
            # Extract coordinates and data
            bbox = obj.bbox.to_xyxy() 
            x1, y1, x2, y2 = map(int, bbox)
            conf = obj.score.value
            cls_name = obj.category.name
            
            # Draw Red Bounding Boxes
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_rgb, f"{cls_name} {conf:.2f}", (x1, max(y1-8, 12)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            defect_data.append({
                "Defect Type": cls_name,
                "Confidence": f"{conf:.1%}",
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
            })

    # 7. Display Status & Image Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Board")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Inspection Result")
        st.image(annotated_rgb, use_container_width=True)

    # 8. Quality Assurance Report
    num_defects = len(result.object_prediction_list)
    st.markdown("---")
    st.subheader("📊 Inspection Report")
    
    if num_defects == 0:
        st.success("✅ **PASS**: No defects detected above the selected confidence threshold.")
    else:
        st.error(f"❌ **FAIL**: {num_defects} defect(s) detected on this board.")
        st.dataframe(defect_data, use_container_width=True)

    # 9. Operator Feedback Loop (Active Learning via Webhook)
    st.markdown("---")
    st.subheader("🔄 Operator Feedback")
    st.markdown("Did the AI make a mistake? Send the image for retraining.")
    
    # Add a text input for the supervisor's context
    operator_note = st.text_input("What is the issue? (Try to use the defect class names on the left for better retraining)")
    
    if st.button("🚩 Report False Positive to QA Channel"):
        with st.spinner("Sending alert..."):
            # 1. Convert the RAW image (for your retraining dataset)
            raw_buffer = io.BytesIO()
            image.save(raw_buffer, format="JPEG")
            raw_bytes = raw_buffer.getvalue()
            
            # 2. Convert the ANNOTATED image (so you can see what the AI saw)
            # Convert OpenCV numpy array back to bytes
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            annotated_bytes = buffer.tobytes()
            
            WEBHOOK_URL = webhook_url = st.secrets["DISCORD_WEBHOOK"].strip().strip("'").strip('"') # Keep your Discord URL here
            
            try:
                # Discord allows sending multiple files in one message
                files = {
                    "file1": (f"raw_board_{datetime.now().strftime('%H%M%S')}.jpg", raw_bytes, "image/jpeg"),
                    "file2": (f"ai_prediction_{datetime.now().strftime('%H%M%S')}.jpg", annotated_bytes, "image/jpeg")
                }
                
                # Format the chat message
                message = "🚨 *AOI Alert:* The supervisor flagged a prediction issue."
                if operator_note:
                    message += f"\n**Supervisor Note:** {operator_note}"
                
                payload = {"content": message}
                
                response = requests.post(WEBHOOK_URL, data=payload, files=files)
                
                if response.status_code in [200, 204]:
                    st.success("✅ Alert, notes, and images successfully sent to the QA team!")
                else:
                    st.warning(f"⚠️ Failed to send to chat. (Status Code: {response.status_code})")
                    
            except Exception as e:
                st.error(f"Network error: {e}")