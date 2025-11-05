import streamlit as st
import os
import subprocess
import json
import pandas as pd
import tempfile
from pathlib import Path
import shutil
import time
from PIL import Image
import numpy as np
import zipfile
import warnings

# Suppress deprecation warnings in UI
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Page config
st.set_page_config(
    page_title="Hand Glove Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {font-size: 24px !important; font-weight: 700 !important;}
    .sub-header {font-size: 20px !important; font-weight: 600 !important;}
    .stButton>button {width: 100%;}
    .stProgress > div > div > div > div {background-color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

# Constants
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
DEFAULT_IMG_SIZE = 640
DEFAULT_MAX_DET = 300

# Session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
    st.session_state.progress = 0
    st.session_state.results = None
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

def reset_state():
    st.session_state.processing = False
    st.session_state.progress = 0
    st.session_state.results = None

def process_images():
    st.session_state.processing = True
    st.session_state.progress = 0
    st.session_state.results = None
    
    try:
        # Create temp dir for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare input directory (temp) and project output directory (persistent)
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = 'output'
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            if st.session_state.input_type == "Upload Images":
                # Handle uploaded files
                for file in st.session_state.uploaded_files:
                    with open(os.path.join(input_dir, file.name), 'wb') as f:
                        f.write(file.getbuffer())
            else:
                # Handle folder input
                folder_path = st.session_state.get('folder_path', '')
                if folder_path and os.path.isdir(folder_path):
                    for file_name in st.session_state.uploaded_files:
                        src_path = os.path.join(folder_path, file_name)
                        dst_path = os.path.join(input_dir, file_name)
                        shutil.copy2(src_path, dst_path)
            
            # Build command with all parameters (some are hidden defaults)
            # Map 'auto' to 'cpu' to avoid invalid CUDA device on machines without GPU
            _device = str(st.session_state.device)
            if _device == 'auto':
                _device = 'cpu'
            cmd = [
                'python', 'main.py',
                '--input', input_dir,
                '--output', output_dir,
                '--confidence', str(st.session_state.confidence),
                '--imgsz', str(st.session_state.imgsz),
                '--iou', str(st.session_state.iou),
                '--max-det', str(st.session_state.max_det),
                '--device', _device,
                '--log-path', 'logs.json'
            ]
            
            if st.session_state.half_precision:
                cmd.append('--half')
            
            if st.session_state.no_annotate:
                cmd.append('--no-annotate')
            
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Process output (do not render raw stdout to UI)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            total_files = len(st.session_state.uploaded_files)
            
            for line in process.stdout:
                if 'Processed ' in line and '/' in line:
                    try:
                        processed = int(line.split('Processed ')[1].split('/')[0])
                        st.session_state.progress = (processed / total_files) * 100
                        progress_bar.progress(int(st.session_state.progress))
                        status_text.text(f"Processing: {processed}/{total_files} images")
                    except:
                        pass
            
            process.wait()
            
            if process.returncode == 0:
                # Load results directly from project logs.json and list images from project output folder
                project_log_path = 'logs.json'
                if os.path.exists(project_log_path):
                    with open(project_log_path, 'r', encoding='utf-8') as f:
                        st.session_state.results = json.load(f)

                # Map output images (append to existing contents, do not delete)
                output_images = {}
                if os.path.exists(output_dir):
                    for img_file in os.listdir(output_dir):
                        if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            output_images[img_file] = os.path.join(output_dir, img_file)

                st.session_state.output_images = output_images
                st.session_state.processing = False
                st.session_state.progress = 100

                # Show success message
                st.success(f"Successfully processed {len(st.session_state.uploaded_files)} images!")
                
            else:
                st.error(f"Processing failed with return code {process.returncode}")
                st.session_state.processing = False
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.processing = False

def main():
    st.title("🖐️ Hand Glove Detection")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Input type selection
        st.subheader("1. Input Type")
        input_type = st.radio(
            "Select input type:",
            ["Upload Images", "Select Folder"],
            key='input_type'
        )
        
        # File or folder input based on selection
        if st.session_state.input_type == "Upload Images":
            uploaded_files = st.file_uploader(
                "Choose images to process",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'],
                accept_multiple_files=True,
                key='uploaded_files'
            )
        else:
            folder_path = st.text_input(
                "Enter folder path containing images:",
                key='folder_path'
            )
            if folder_path and os.path.isdir(folder_path):
                st.session_state.uploaded_files = [f for f in os.listdir(folder_path) 
                                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))]
            else:
                st.session_state.uploaded_files = []
        
        # Model settings - only confidence threshold
        st.subheader("2. Detection Settings")
        st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            key='confidence',
            help="Minimum confidence score for detections (0.1 to 1.0)"
        )
        
        # Hidden defaults
        st.session_state.iou = DEFAULT_IOU
        st.session_state.imgsz = DEFAULT_IMG_SIZE
        st.session_state.max_det = DEFAULT_MAX_DET
        st.session_state.device = 'auto'
        st.session_state.half_precision = False
        st.session_state.no_annotate = False
        
        # Process button
        st.markdown("---")
        process_btn = st.button(
            "🚀 Process Images",
            disabled=not st.session_state.get('uploaded_files') or st.session_state.processing,
            on_click=process_images,
            type="primary"
        )
    
    # Main content area
    if not st.session_state.get('uploaded_files'):
        # Show instructions if no files uploaded
        st.info("👈 Select input type and provide images using the sidebar to get started!")
        
        # Example section
        with st.expander("How to use"):
            st.markdown("""
            1. **Choose Input** - Select between uploading images or providing a folder path
            2. **Set Confidence** - Adjust the confidence threshold (0.1 to 1.0)
            3. **Process** - Click the 'Process Images' button
            4. **View Results** - See detections and download results
            
            ### Tips:
            - For better accuracy, use higher resolution images
            - Lower confidence threshold for more detections (but possibly more false positives)
            - Higher confidence for more reliable detections (but might miss some objects)
            """)
    
    # Show processing status
    if st.session_state.processing:
        st.sidebar.info("⏳ Processing images...")
    
    # Show results if available
    if 'results' in st.session_state and st.session_state.results is not None:
        st.subheader("📊 Detection Results")
        
        # Results summary
        total_detections = sum(len(img['detections']) for img in st.session_state.results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Images Processed", len(st.session_state.results))
        with col2:
            st.metric("Total Detections", total_detections)
        with col3:
            st.metric("Avg. Detections/Image", 
                     f"{total_detections/len(st.session_state.results):.1f}" if st.session_state.results else 0)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📋 Image List", "📊 All Detections"])
        
        with tab1:
            # Desktop-like file grid and detail view
            st.subheader("🖼️ Processed Images")

            # Grid of tiles (filenames as icons) with Open buttons
            cols_per_row = 5
            cols = st.columns(cols_per_row)

            sorted_results = sorted(st.session_state.results, key=lambda x: x['filename'])
            for idx, img in enumerate(sorted_results):
                img_name = img['filename']
                if img_name in st.session_state.output_images:
                    with cols[idx % cols_per_row]:
                        tile = st.container()
                        with tile:
                            # Small thumbnail to mimic desktop icon
                            thumb_path = st.session_state.output_images[img_name]
                            try:
                                thumb = Image.open(thumb_path).copy()
                                thumb.thumbnail((160, 120))
                                st.image(thumb, use_container_width=True)
                            except Exception:
                                st.caption("[Image]")

                            st.caption(img_name)
                            if st.button("Open", key=f"open_{img_name}"):
                                st.session_state.selected_image = img_name

            st.markdown("---")

            # Detail pane for the selected image
            if st.session_state.selected_image:
                sel_name = st.session_state.selected_image
                if sel_name in st.session_state.output_images:
                    left, right = st.columns([2, 1])
                    with left:
                        try:
                            full_img = Image.open(st.session_state.output_images[sel_name]).copy()
                            # Resize to exactly 640x320 as requested
                            full_img = full_img.resize((640, 320))
                            st.image(full_img, caption=f"{sel_name} (640x320)")
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    with right:
                        # Show JSON for just this image
                        record = next((r for r in st.session_state.results if r['filename'] == sel_name), None)
                        if record is not None:
                            st.subheader("JSON")
                            st.json(record, expanded=False)
                        else:
                            st.info("No JSON found for selected image.")
        
        with tab2:
            # Original detailed view
            if total_detections > 0:
                st.subheader("📈 All Detections")
                
                # Create dataframe for all detections
                detections_list = []
                for img in st.session_state.results:
                    for det in img['detections']:
                        detections_list.append({
                            'Image': img['filename'],
                            'Label': det['label'],
                            'Confidence': det['confidence'],
                            'BBox X1': det['bbox'][0],
                            'BBox Y1': det['bbox'][1],
                            'BBox X2': det['bbox'][2],
                            'BBox Y2': det['bbox'][3],
                            'Width': det['bbox'][2] - det['bbox'][0],
                            'Height': det['bbox'][3] - det['bbox'][1]
                        })
                
                detections_df = pd.DataFrame(detections_list)
                
                # Show detections table with better formatting
                st.dataframe(
                    detections_df,
                    column_config={
                        'Image': st.column_config.TextColumn("Image"),
                        'Label': st.column_config.TextColumn("Label"),
                        'Confidence': st.column_config.ProgressColumn(
                            "Confidence",
                            format="%.2f",
                            min_value=0,
                            max_value=1
                        ),
                        'Width': st.column_config.NumberColumn("Width (px)", format="%d"),
                        'Height': st.column_config.NumberColumn("Height (px)", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No detections found in any of the images.")
        
        # Download buttons at the bottom
        st.download_button(
            label="📥 Download All Results (JSON)",
            data=json.dumps(st.session_state.results, indent=2),
            file_name="detection_results.json",
            mime="application/json"
        )
        
        if 'output_images' in st.session_state and st.session_state.output_images:
            # Create a zip of output images
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, 'detection_results.zip')
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for img_name, img_path in st.session_state.output_images.items():
                        zipf.write(img_path, os.path.basename(img_path))
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download All Annotated Images (ZIP)",
                        data=f,
                        file_name="detection_results.zip",
                        mime="application/zip"
                    )

if __name__ == "__main__":
    main()
