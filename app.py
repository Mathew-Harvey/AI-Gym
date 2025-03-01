import streamlit as st
import os
import json
import random
from PIL import Image
import numpy as np
import time

# Set page configuration for a cleaner look
st.set_page_config(
    page_title="AI Training Gym",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a loading message
st.info("Loading application components... This may take a moment on free tier hosting.")

# Import dependencies directly without caching
try:
    import cv2
    st.success("OpenCV loaded successfully!")
except ImportError:
    st.error("Failed to load OpenCV. Some functionality may be limited.")
    cv2 = None

try:
    from ultralytics import YOLO
    st.success("YOLO loaded successfully!")
    YOLO_available = True
except ImportError:
    st.warning("YOLO not available. Training and inference will be disabled.")
    YOLO = None
    YOLO_available = False

# Import drawable canvas with error handling
try:
    from streamlit_drawable_canvas import st_canvas
    canvas_available = True
except ImportError:
    st.warning("Drawing canvas not available. Annotation will be limited.")
    canvas_available = False
    # Fallback if the component fails to load
    def st_canvas(*args, **kwargs):
        st.error("Drawing component failed to load. Please refresh the page or try again later.")
        return None

# Add custom CSS for better styling - optimized and simplified
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 1rem;}
    .step-header {font-size: 1.6rem; font-weight: 600; color: #0D47A1; margin: 1rem 0;}
    .sub-header {font-size: 1.2rem; font-weight: 500; color: #0277BD; margin: 0.8rem 0;}
    .hint-text {font-size: 0.9rem; color: #455A64; font-style: italic;}
    .info-box {background-color: #E3F2FD; padding: 0.8rem; border-radius: 0.4rem; border-left: 0.4rem solid #1E88E5; margin: 0.8rem 0;}
    .success-box {background-color: #E8F5E9; padding: 0.8rem; border-radius: 0.4rem; border-left: 0.4rem solid #43A047; margin: 0.8rem 0;}
    .warning-box {background-color: #FFF8E1; padding: 0.8rem; border-radius: 0.4rem; border-left: 0.4rem solid #FFA000; margin: 0.8rem 0;}
    .error-box {background-color: #FFEBEE; padding: 0.8rem; border-radius: 0.4rem; border-left: 0.4rem solid #E53935; margin: 0.8rem 0;}
    .step-indicator {font-size: 1rem; font-weight: 500; color: white; background-color: #1E88E5; padding: 0.2rem 0.6rem; border-radius: 2rem; margin-right: 0.5rem;}
    .progress-container {margin: 1.5rem 0;}
</style>
""", unsafe_allow_html=True)

# Create necessary directories
for dir in ["dataset/images", "dataset/labels", "annotations", "models", "output"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

def extract_frames(video_path, output_dir, fps=1, max_frames=50):
    """Extract frames from a video at a specified frame rate with a maximum limit for performance."""
    if cv2 is None:
        st.error("OpenCV is not available. Cannot extract frames from video.")
        return 0
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps) if fps > 0 else 1
    extracted_frames = 0
    
    # Show a progress bar
    progress_bar = st.progress(0)
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while success and extracted_frames < max_frames:
        # Update progress every 10 frames
        if count % 10 == 0 and video_length > 0:
            progress_bar.progress(min(1.0, count / video_length))
            
        if count % interval == 0:
            frame_name = f"{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), image)
            extracted_frames += 1
            
        success, image = vidcap.read()
        count += 1
        
    vidcap.release()
    progress_bar.empty()  # Remove progress bar when done
    return extracted_frames

# App Header
st.markdown('<div class="main-header">🤖 AI Training Gym</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Welcome to the AI Training Gym! This tool helps you build your own custom object detection model in 5 simple steps:
<ol>
    <li><strong>Upload Data</strong>: Add images or videos that contain the objects you want to detect</li>
    <li><strong>Define Classes</strong>: Specify what types of objects you want to identify</li>
    <li><strong>Annotate Images</strong>: Draw boxes around objects in your images</li>
    <li><strong>Train Model</strong>: Train your AI to recognize these objects</li>
    <li><strong>Test Model</strong>: Upload new images/videos to see your AI in action!</li>
</ol>
The sidebar navigation guides you through each step in order. This process mimics how professional AI models are built, but made simple!
</div>
""", unsafe_allow_html=True)

# Sidebar navigation with progress tracking
menu = ["Home", "Class Definition", "Annotation", "Training", "Inference"]
icons = ["🏠", "🏷️", "✏️", "🧠", "🔍"]
descriptions = [
    "Upload your data",
    "Define object classes",
    "Annotate your images",
    "Train your model", 
    "Test your model"
]

# Track progress through workflow
if 'workflow_progress' not in st.session_state:
    st.session_state.workflow_progress = 0

# Sidebar header
st.sidebar.markdown('<div class="sub-header">Navigation</div>', unsafe_allow_html=True)

# Create a visually appealing sidebar navigation
choice = None
for i, (section, icon, desc) in enumerate(zip(menu, icons, descriptions)):
    # Mark completed steps
    completed = "✅ " if i < st.session_state.workflow_progress else ""
    current = "→ " if i == st.session_state.workflow_progress else ""
    
    # Create a button-like effect for each menu item
    if st.sidebar.button(f"{completed}{current}{icon} {section}: {desc}", key=f"nav_{i}", 
                         use_container_width=True, 
                         type="primary" if i == st.session_state.workflow_progress else "secondary"):
        choice = section

# Default to the current progress step if no choice is made
if choice is None:
    choice = menu[st.session_state.workflow_progress]

# Add workflow progress visualization
st.sidebar.markdown('<div class="progress-container">', unsafe_allow_html=True)
st.sidebar.progress(st.session_state.workflow_progress / (len(menu) - 1))
st.sidebar.markdown(f"Progress: Step {st.session_state.workflow_progress + 1} of {len(menu)}", unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Add help information in the sidebar
with st.sidebar.expander("❓ Need Help?"):
    st.markdown("""
    ### Common Questions
    
    **What is object detection?**  
    Object detection is an AI technique that identifies and locates objects within images or videos.
    
    **What is YOLO?**  
    YOLO (You Only Look Once) is a popular and efficient object detection algorithm.
    
    **How much data do I need?**  
    Start with at least 50 images per class for basic models. More images usually mean better results!
    
    **How long does training take?**  
    Training typically takes 15-30 minutes on a decent computer, depending on your dataset size.
    
    **Can I save my progress?**  
    Yes! All your uploads, annotations, and models are saved automatically.
    """)

# Home Section: Upload images or videos
if choice == "Home":
    st.markdown('<div class="step-header"><span class="step-indicator">1</span>Upload Training Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>What to do here:</strong> Upload images or videos containing objects you want your AI to recognize.
        
        <strong>Tips:</strong>
        <ul>
            <li>Include a variety of backgrounds, lighting conditions, and angles</li>
            <li>For videos, we'll extract frames at your chosen rate</li>
            <li>JPG and PNG images, MP4 videos are supported</li>
            <li>Aim for at least 50 images per object class</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload images or videos", 
            accept_multiple_files=True, 
            type=["jpg", "jpeg", "png", "mp4"]
        )
        
        fps = st.slider(
            "Frame extraction rate (frames per second)", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0,
            help="Higher values extract more frames from videos"
        )
        
        if st.button("Process Uploads", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Processing your files..."):
                    img_count = 0
                    vid_count = 0
                    frames_count = 0
                    
                    progress_bar = st.progress(0)
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress_bar.progress((i) / len(uploaded_files))
                        file_path = os.path.join("dataset/images", uploaded_file.name)
                        
                        if uploaded_file.type.startswith("image"):
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            img_count += 1
                            
                        elif uploaded_file.type.startswith("video"):
                            temp_path = "temp_video.mp4"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            frames = extract_frames(temp_path, "dataset/images", fps)
                            frames_count += frames
                            vid_count += 1
                            os.remove(temp_path)
                    
                    progress_bar.progress(1.0)
                    
                    # Show success message with stats
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ Data processed successfully!
                    <ul>
                        <li>{img_count} images uploaded directly</li>
                        <li>{vid_count} videos processed</li>
                        <li>{frames_count} frames extracted from videos</li>
                        <li>Total: {img_count + frames_count} images in your dataset</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update progress if first step complete
                    if st.session_state.workflow_progress == 0:
                        st.session_state.workflow_progress = 1
                    
                    # Suggest next steps
                    st.markdown("""
                    <div class="info-box">
                    <strong>Next Step:</strong> Go to the "Class Definition" section to define what objects you want to detect.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                ⚠️ Please upload at least one file to continue.
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Dataset Status</div>', unsafe_allow_html=True)
        
        # Count images in the dataset
        image_count = len([f for f in os.listdir("dataset/images") if f.endswith((".jpg", ".jpeg", ".png"))])
        
        if image_count > 0:
            st.markdown(f"""
            <div class="success-box">
            ✅ Your dataset contains {image_count} images
            </div>
            """, unsafe_allow_html=True)
            
            # Show a single sample image to reduce load on slow servers
            st.markdown('<div class="sub-header">Sample Image</div>', unsafe_allow_html=True)
            images = [f for f in os.listdir("dataset/images") if f.endswith((".jpg", ".jpeg", ".png"))]
            if images:
                sample_img = random.choice(images)
                img_path = os.path.join("dataset/images", sample_img)
                st.image(img_path, caption=f"Random sample: {sample_img}", use_column_width=True, width=300)
        else:
            st.markdown("""
            <div class="warning-box">
            ⚠️ No images in dataset yet.
            <br>
            Upload some images or videos to get started!
            </div>
            """, unsafe_allow_html=True)

# Class Definition Section: Define object classes
elif choice == "Class Definition":
    st.markdown('<div class="step-header"><span class="step-indicator">2</span>Define Classes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>What to do here:</strong> Define the types of objects you want your AI to detect.
        
        <strong>Examples:</strong>
        <ul>
            <li>For a wildlife detection system: "deer", "bear", "fox"</li>
            <li>For a traffic monitoring system: "car", "truck", "motorcycle", "pedestrian"</li>
            <li>For quality control: "defect_type_1", "defect_type_2", "normal"</li>
        </ul>
        
        <strong>Tips:</strong>
        <ul>
            <li>Be specific and consistent in your naming</li>
            <li>Avoid spaces in class names (use underscores instead)</li>
            <li>Start with 2-5 classes for your first model</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if classes file exists and load it
        default_classes = ""
        if os.path.exists("classes.txt"):
            with open("classes.txt") as f:
                default_classes = f.read()
        
        classes_text = st.text_area(
            "Enter class names (one per line)", 
            value=default_classes,
            placeholder="car\npedestrian\nbicycle",
            height=200,
            help="Each line represents one type of object your model will detect"
        )
        
        if st.button("Save Classes", use_container_width=True, type="primary"):
            if classes_text.strip():
                # Check for invalid class names
                class_list = [c.strip() for c in classes_text.splitlines() if c.strip()]
                invalid_classes = [c for c in class_list if ' ' in c]
                
                if invalid_classes:
                    st.markdown(f"""
                    <div class="error-box">
                    ❌ The following class names contain spaces, which isn't recommended:
                    <ul>
                    {"".join([f"<li>{c}</li>" for c in invalid_classes])}
                    </ul>
                    Consider replacing spaces with underscores (e.g., "stop_sign" instead of "stop sign").
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with open("classes.txt", "w") as f:
                        f.write(classes_text)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ {len(class_list)} classes saved successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Update progress if second step complete
                    if st.session_state.workflow_progress == 1:
                        st.session_state.workflow_progress = 2
                    
                    # Suggest next steps
                    st.markdown("""
                    <div class="info-box">
                    <strong>Next Step:</strong> Proceed to the "Annotation" section to start labeling your images.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                ❌ Please enter at least one class name.
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Classes Status</div>', unsafe_allow_html=True)
        
        if os.path.exists("classes.txt"):
            with open("classes.txt") as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                
            if classes:
                st.markdown(f"""
                <div class="success-box">
                ✅ You have defined {len(classes)} classes:
                <ul>
                {"".join([f"<li>{c}</li>" for c in classes])}
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                ⚠️ Your classes file exists but is empty.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            ⚠️ No classes defined yet.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Best Practices:</strong>
        <ul>
            <li>Use clear, descriptive names</li>
            <li>Be consistent with naming conventions</li>
            <li>Consider hierarchical naming for related objects (car_sedan, car_suv)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Annotation Section: Annotate images with bounding boxes
elif choice == "Annotation":
    st.markdown('<div class="step-header"><span class="step-indicator">3</span>Annotate Images</div>', unsafe_allow_html=True)
    
    if not os.path.exists("classes.txt") or not os.path.getsize("classes.txt") > 0:
        st.markdown("""
        <div class="error-box">
        ❌ Please define classes first in the "Class Definition" section before annotating images.
        </div>
        """, unsafe_allow_html=True)
    else:
        with open("classes.txt") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        
        images = [f for f in os.listdir("dataset/images") if f.endswith((".jpg", ".jpeg", ".png"))]
        
        if not images:
            st.markdown("""
            <div class="error-box">
            ❌ No images found. Please upload data in the "Home" section before annotating.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <strong>What to do here:</strong> Draw bounding boxes around objects in your images and assign them to the correct class.
            
            <strong>Instructions:</strong>
            <ol>
                <li>Select an image from the dropdown menu</li>
                <li>Draw a rectangle around an object by clicking and dragging</li>
                <li>Select the correct class for that object</li>
                <li>Click "Add Box" to save the annotation</li>
                <li>Repeat for all objects in the image</li>
                <li>Click "Save Annotations" when finished with the current image</li>
                <li>Move to the next image and repeat</li>
            </ol>
            
            <strong>Tips:</strong>
            <ul>
                <li>Draw tight boxes around objects for better training results</li>
                <li>Include the entire object within the box</li>
                <li>Annotate all instances of your classes in each image</li>
                <li>Try to annotate at least 50 examples of each class</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Display annotation progress
            total_images = len(images)
            annotated_images = len([f for f in os.listdir("annotations") if f.endswith(".json")])
            progress_percentage = int((annotated_images / total_images) * 100) if total_images > 0 else 0
            
            st.markdown(f"""
            <div class="info-box">
            <strong>Annotation Progress:</strong> {annotated_images}/{total_images} images ({progress_percentage}%)
            </div>
            """, unsafe_allow_html=True)
            
            # Organize layout with columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Add sorting and filtering for images
                sort_options = ["Alphabetical", "Recently Added", "Not Annotated First"]
                sort_choice = st.selectbox("Sort Images By:", sort_options)
                
                # Apply sorting
                if sort_choice == "Recently Added":
                    images.sort(key=lambda x: os.path.getmtime(os.path.join("dataset/images", x)), reverse=True)
                elif sort_choice == "Not Annotated First":
                    # Find which images have annotations
                    annotated = [f.replace(".json", "") for f in os.listdir("annotations") if f.endswith(".json")]
                    not_annotated = [img for img in images if img not in annotated]
                    annotated_imgs = [img for img in images if img in annotated]
                    images = not_annotated + annotated_imgs
                else:  # Alphabetical
                    images.sort()
                
                # Add search filter
                search = st.text_input("Search images:", "", placeholder="Type to filter images...")
                if search:
                    images = [img for img in images if search.lower() in img.lower()]
                
                # Image selection with thumbnails
                img_select = st.selectbox("Select Image to Annotate:", images)
                img_path = os.path.join("dataset/images", img_select)
                img = Image.open(img_path)
                width, height = img.size
                
                # Display image dimensions
                st.markdown(f"<div class='hint-text'>Image dimensions: {width} × {height} pixels</div>", unsafe_allow_html=True)
                
                # Load existing annotations
                annotation_file = os.path.join("annotations", img_select + ".json")
                if os.path.exists(annotation_file):
                    with open(annotation_file) as f:
                        annotation_data = json.load(f)
                        st.session_state.annotations = annotation_data["boxes"]
                        st.info(f"Loaded {len(st.session_state.annotations)} existing annotations.")
                else:
                    st.session_state.annotations = []
                
                # Canvas for drawing with error handling for slow servers
                if not canvas_available:
                    st.error("Drawing canvas component is not available. Please check your installation.")
                else:
                    try:
                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
                            stroke_width=2,
                            background_image=img,
                            update_streamlit=True,
                            height=min(500, height),  # Reduced height to decrease load
                            width=min(700, width),    # Reduced width to decrease load
                            drawing_mode="rect",
                            key=f"canvas_{img_select}",
                        )
                    except Exception as e:
                        st.error(f"Drawing canvas failed to load. Try refreshing the page. Error: {str(e)}")
                        canvas_result = None
            
            with col2:
                st.markdown('<div class="sub-header">Box Controls</div>', unsafe_allow_html=True)
                
                # Class selection with color coding
                class_colors = ["#FF5722", "#2196F3", "#4CAF50", "#9C27B0", "#FFC107", "#607D8B", "#E91E63", "#00BCD4"]
                class_index = st.selectbox(
                    "Select Class for New Box:",
                    range(len(classes)),
                    format_func=lambda i: classes[i]
                )
                class_label = classes[class_index]
                
                # Show color swatch for the selected class
                color = class_colors[class_index % len(class_colors)]
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px;">
                Selected: {class_label}
                </div>
                """, unsafe_allow_html=True)
                
                # Add box button
                if st.button("Add Box", use_container_width=True, type="primary"):
                    if canvas_result is not None and canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        if objects:
                            last_object = objects[-1]
                            if last_object["type"] == "rect":
                                box = {
                                    "class": class_label,
                                    "left": last_object["left"],
                                    "top": last_object["top"],
                                    "width": last_object["width"],
                                    "height": last_object["height"],
                                    "color": color
                                }
                                st.session_state.annotations.append(box)
                                st.success(f"Added box for '{class_label}'.")
                                # Force a rerun to update the displayed annotations
                                st.experimental_rerun()
                        else:
                            st.warning("Please draw a box on the image first.")
                    else:
                        st.warning("Canvas not available. Please try refreshing the page.")
                
                # Save annotations
                if st.button("Save All Annotations", use_container_width=True):
                    annotation_data = {
                        "image": img_select,
                        "width": width,
                        "height": height,
                        "boxes": st.session_state.annotations
                    }
                    os.makedirs("annotations", exist_ok=True)
                    with open(annotation_file, "w") as f:
                        json.dump(annotation_data, f)
                    
                    st.success(f"Saved {len(st.session_state.annotations)} annotations for {img_select}")
                    
                    # Update progress if needed
                    if st.session_state.workflow_progress == 2 and annotated_images + 1 >= min(10, total_images):
                        st.session_state.workflow_progress = 3
                
                # Skip to next unannotated image
                if st.button("Skip to Next Unannotated", use_container_width=True):
                    unannotated = [img for img in images if not os.path.exists(os.path.join("annotations", img + ".json"))]
                    if unannotated and unannotated[0] != img_select:
                        # We need to set a session state to change the selectbox value
                        st.session_state.next_image = unannotated[0]
                        st.experimental_rerun()
                    else:
                        st.info("No more unannotated images!")
                
                # Display current annotations
                st.markdown('<div class="sub-header">Current Annotations</div>', unsafe_allow_html=True)
                
                if st.session_state.annotations:
                    for i, box in enumerate(st.session_state.annotations):
                        box_color = box.get("color", "#FF5722")  # Default color if not specified
                        
                        # Create a colored box display with delete button
                        st.markdown(f"""
                        <div style="background-color: {box_color}20; padding: 8px; border-radius: 4px; 
                                    border-left: 4px solid {box_color}; margin-bottom: 8px;">
                            <strong>{i+1}. {box['class']}</strong><br>
                            <small>Position: ({box['left']:.1f}, {box['top']:.1f})<br>
                            Size: {box['width']:.1f} × {box['height']:.1f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"Delete Box #{i+1}", key=f"del_{img_select}_{i}", use_container_width=True):
                            st.session_state.annotations.pop(i)
                            st.experimental_rerun()
                else:
                    st.markdown("""
                    <div class="hint-text">
                    No annotations for this image yet. Draw boxes around objects and add them.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Annotation tips
                with st.expander("Annotation Tips"):
                    st.markdown("""
                    - Draw boxes that tightly fit around the object
                    - Include the entire object in the box
                    - Annotate all instances of your classes in each image
                    - If an object is partially visible, still annotate it
                    - Be consistent with your annotations across images
                    """)

# Training Section: Train the YOLOv5 model
elif choice == "Training":
    st.markdown('<div class="step-header"><span class="step-indicator">4</span>Train Your Model</div>', unsafe_allow_html=True)
    
    # Verify requirements for training
    requirements_met = True
    error_messages = []
    
    # Check for classes
    if not os.path.exists("classes.txt") or not os.path.getsize("classes.txt") > 0:
        requirements_met = False
        error_messages.append("- No classes defined. Please visit the 'Class Definition' section first.")
    else:
        with open("classes.txt") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # Check for images
    images = [f for f in os.listdir("dataset/images") if f.endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        requirements_met = False
        error_messages.append("- No images found. Please upload images in the 'Home' section.")
    
    # Check for annotations
    annotation_files = [f for f in os.listdir("annotations") if f.endswith(".json")]
    if not annotation_files:
        requirements_met = False
        error_messages.append("- No annotations found. Please annotate your images in the 'Annotation' section.")
    
    if not requirements_met:
        st.markdown("""
        <div class="error-box">
        ❌ Cannot start training. Please fix the following issues:
        <ul>
        """ + "".join([f"<li>{msg}</li>" for msg in error_messages]) + """
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        <strong>What to do here:</strong> Configure and train your custom object detection model.
        
        <strong>What happens during training:</strong>
        <ol>
            <li>Your annotations are converted to YOLO format</li>
            <li>Your dataset is split into training and validation sets</li>
            <li>A YOLOv5 model is trained on your data</li>
            <li>Training progress and performance metrics are displayed</li>
            <li>The best model is saved for later use</li>
        </ol>
        
        <strong>Tips:</strong>
        <ul>
            <li>Training may take 15-30 minutes or more depending on your dataset size</li>
            <li>A good starting point is 50-100 epochs</li>
            <li>You can train multiple models with different settings to compare results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Training configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Training Configuration</div>', unsafe_allow_html=True)
            
            model_size = st.selectbox(
                "Model Size:", 
                ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                index=1,
                help="Larger models are more accurate but slower and require more memory"
            )
            
            model_name = st.text_input(
                "Model Name:", 
                value=f"custom_model_{len(classes)}_classes",
                help="A unique name for this training run"
            )
            
            epochs = st.slider(
                "Training Epochs:", 
                min_value=10, 
                max_value=300, 
                value=50,
                help="More epochs = more training time but potentially better results"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                batch_size = st.select_slider(
                    "Batch Size:", 
                    options=[1, 2, 4, 8, 16, 32, 64],
                    value=8,
                    help="Higher values use more memory but train faster"
                )
                
                img_size = st.select_slider(
                    "Image Size:", 
                    options=[320, 416, 512, 640, 736],
                    value=640,
                    help="Input resolution - higher values are more accurate but slower"
                )
                
                patience = st.slider(
                    "Early Stopping Patience:", 
                    min_value=0, 
                    max_value=50, 
                    value=15,
                    help="Stop training if no improvement after this many epochs"
                )
                
                split_ratio = st.slider(
                    "Train/Val Split Ratio:", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=0.8,
                    help="Percentage of images used for training vs. validation"
                )
        
        with col2:
            st.markdown('<div class="sub-header">Dataset Summary</div>', unsafe_allow_html=True)
            
            # Dataset statistics
            image_count = len(images)
            annotation_count = len(annotation_files)
            
            st.markdown(f"""
            <div class="info-box">
            <strong>Dataset Size:</strong> {image_count} images<br>
            <strong>Annotated Images:</strong> {annotation_count} ({int(annotation_count/image_count*100) if image_count else 0}%)<br>
            <strong>Classes:</strong> {len(classes)}<br>
            </div>
            """, unsafe_allow_html=True)
            
            # Class distribution
            class_counts = {c: 0 for c in classes}
            for annot_file in annotation_files:
                with open(os.path.join("annotations", annot_file)) as f:
                    data = json.load(f)
                    for box in data.get("boxes", []):
                        class_name = box.get("class", "")
                        if class_name in class_counts:
                            class_counts[class_name] += 1
            
            st.markdown('<div class="sub-header">Class Distribution</div>', unsafe_allow_html=True)
            for cls, count in class_counts.items():
                percentage = int((count / sum(class_counts.values())) * 100) if sum(class_counts.values()) else 0
                st.markdown(f"""
                <div style="margin-bottom: 5px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{cls}</div>
                        <div>{count} ({percentage}%)</div>
                    </div>
                    <div style="background-color: #e0e0e0; border-radius: 5px; height: 10px; margin-top: 2px;">
                        <div style="background-color: #1E88E5; width: {percentage}%; height: 10px; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Previous training runs
            model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
            if model_dirs:
                st.markdown('<div class="sub-header">Previous Training Runs</div>', unsafe_allow_html=True)
                for model_dir in model_dirs:
                    st.markdown(f"- {model_dir}")
        
        # Training button
        if st.button("Start Training", use_container_width=True, type="primary"):
            with st.spinner("Preparing for training..."):
                # Convert annotations to YOLO format
                with st.status("Converting annotations to YOLO format...") as status:
                    for img in images:
                        annotation_file = os.path.join("annotations", img + ".json")
                        label_file = os.path.join("dataset/labels", img.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt"))
                        if os.path.exists(annotation_file):
                            with open(annotation_file) as f:
                                annotation_data = json.load(f)
                            width = annotation_data["width"]
                            height = annotation_data["height"]
                            boxes = annotation_data["boxes"]
                            with open(label_file, "w") as lf:
                                for box in boxes:
                                    class_name = box["class"]
                                    class_id = classes.index(class_name)
                                    left = box["left"]
                                    top = box["top"]
                                    box_width = box["width"]
                                    box_height = box["height"]
                                    x_center = (left + box_width / 2) / width
                                    y_center = (top + box_height / 2) / height
                                    norm_width = box_width / width
                                    norm_height = box_height / height
                                    lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                        else:
                            # Create empty label file for images without annotations
                            open(label_file, "a").close()
                    status.update(label="✅ Annotations converted", state="complete")

                # Split dataset
                with st.status("Splitting dataset into train/validation sets...") as status:
                    all_images = [os.path.join("dataset/images", f) for f in images]
                    random.shuffle(all_images)
                    split_index = int(split_ratio * len(all_images))
                    train_images = all_images[:split_index]
                    val_images = all_images[split_index:] if split_index < len(all_images) else train_images[:max(1, int(len(train_images) * 0.2))]

                    with open("dataset/train.txt", "w") as f:
                        for img in train_images:
                            f.write(f"{os.path.abspath(img)}\n")
                    with open("dataset/val.txt", "w") as f:
                        for img in val_images:
                            f.write(f"{os.path.abspath(img)}\n")
                    status.update(label=f"✅ Split dataset: {len(train_images)} training, {len(val_images)} validation images", state="complete")

                # Create dataset.yaml
                with st.status("Creating YAML configuration...") as status:
                    with open("dataset.yaml", "w") as f:
                        f.write(f"train: {os.path.abspath('dataset/train.txt')}\n")
                        f.write(f"val: {os.path.abspath('dataset/val.txt')}\n")
                        f.write(f"nc: {len(classes)}\n")
                        f.write(f"names: {classes}\n")
                    status.update(label="✅ Created YAML configuration", state="complete")

                # Start training
                st.markdown("""
                <div class="info-box">
                <strong>Training Started!</strong><br>
                The model is now training. This process may take 15-30 minutes or more depending on your dataset size and settings.
                </div>
                """, unsafe_allow_html=True)
                
                # Train the model
                progress_bar = st.progress(0)
                training_status = st.empty()
                
                try:
                    # Check if YOLO is available
                    if not YOLO_available:
                        st.error("YOLO is not available. Cannot train the model.")
                    else:
                        # Simulate initial loading for better UX on slow servers
                        for i in range(10):
                            progress_bar.progress(i/20)
                            training_status.info(f"Initializing model and preparing dataset...")
                            time.sleep(0.2)  # Short delay for UX
                        
                        # Start actual training
                        model = YOLO('yolov5s.pt')  # Start with small YOLOv5 model
                        results = model.train(
                            data='dataset.yaml',
                            epochs=epochs,
                            imgsz=img_size,
                            batch=batch_size,
                            patience=patience,
                            project='models',
                            name=model_name
                        )
                        
                        # Update progress if fourth step complete
                        if st.session_state.workflow_progress == 3:
                            st.session_state.workflow_progress = 4
                        
                        st.markdown(f"""
                        <div class="success-box">
                        ✅ Training completed! Model saved in 'models/{model_name}'
                        
                        <strong>Next Step:</strong> Go to the "Inference" section to test your model.
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    results = None
                    
                progress_bar.empty()
                training_status.empty()

# Inference Section: Test the trained model
elif choice == "Inference":
    st.markdown('<div class="step-header"><span class="step-indicator">5</span>Test Your Model</div>', unsafe_allow_html=True)
    
    model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    
    if not model_dirs:
        st.markdown("""
        <div class="error-box">
        ❌ No trained models found. Please go to the "Training" section to train a model first.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        <strong>What to do here:</strong> Upload new images or videos to see how well your model detects objects.
        
        <strong>Tips:</strong>
        <ul>
            <li>Use images that weren't part of your training set for a fair evaluation</li>
            <li>Try different confidence thresholds to control detection sensitivity</li>
            <li>Compare results across multiple models if you've trained more than one</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Test Your Model</div>', unsafe_allow_html=True)
            
            # Model selection with metadata
            model_select = st.selectbox(
                "Select Model:", 
                model_dirs,
                help="Choose from your trained models"
            )
            
            # Display model details if available
            model_path = os.path.join("models", model_select)
            if os.path.exists(os.path.join(model_path, "args.yaml")):
                try:
                    import yaml
                    with open(os.path.join(model_path, "args.yaml")) as f:
                        model_args = yaml.safe_load(f)
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>Model Details:</strong><br>
                    Base: {model_args.get('model', 'N/A')}<br>
                    Epochs: {model_args.get('epochs', 'N/A')}<br>
                    Image Size: {model_args.get('imgsz', 'N/A')}<br>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    pass
            
            # Inference settings
            confidence = st.slider(
                "Confidence Threshold:", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.4,
                help="Lower values show more detections with higher false positive rate"
            )
            
            test_file = st.file_uploader(
                "Upload Test File", 
                type=["jpg", "jpeg", "png", "mp4"],
                help="Upload an image or video to test your model"
            )
            
            # Option to use webcam
            use_webcam = st.checkbox("Use Webcam for Live Detection")
            
            # Inference button
            detect_button = st.button("Detect Objects", use_container_width=True, type="primary")
            
            if detect_button:
                if test_file is not None or use_webcam:
                    with st.spinner("Running detection..."):
                        # Check model existence
                        model_weights_path = os.path.join("models", model_select, "weights", "best.pt")
                        if not os.path.exists(model_weights_path):
                            st.error(f"Model weights not found at {model_weights_path}")
                        else:
                            # Save test file
                            test_extension = os.path.splitext(test_file.name)[1]
                            test_file_path = os.path.join("output", f"test_file{test_extension}")
                            with open(test_file_path, "wb") as f:
                                f.write(test_file.getbuffer())
                            
                            # Show loading indicator for slow servers
                            with st.spinner("Loading YOLO model and running inference... This can take some time on free tier hosting."):
                                # Load model and perform inference
                                model_path = os.path.join("models", model_select, "weights", "best.pt")
                                YOLO_class = load_yolo()
                                if YOLO_class is None:
                                    st.error("Failed to load YOLO. Please refresh and try again.")
                                    results = None
                                else:
                                    model = YOLO_class(model_path)
                                    results = model(test_file_path, save=True, project="output", name="test")
                            
                            # Add a small delay to ensure files are written
                            time.sleep(2)
                            
                            # Display results
                            if test_file.type.startswith("image"):
                                output_img = os.path.join("output", "test", os.path.basename(test_file_path))
                                if os.path.exists(output_img):
                                    st.image(output_img, caption="Detection Result", use_column_width=True)
                                    
                                    # Extract detection results for display
                                    detections = []
                                    if results is not None:
                                        for r in results:
                                            boxes = r.boxes
                                            for box in boxes:
                                                cls = int(box.cls[0])
                                                class_name = model.names[cls]
                                                conf = float(box.conf[0])
                                                detections.append((class_name, conf))
                                    
                                    if detections:
                                        st.markdown('<div class="sub-header">Detections</div>', unsafe_allow_html=True)
                                        for i, (class_name, conf) in enumerate(detections):
                                            st.markdown(f"{i+1}. {class_name}: {conf:.2f} confidence")
                                    else:
                                        st.info("No objects detected in this image.")
                                else:
                                    st.error("Failed to generate output image.")
                            
                            elif test_file.type.startswith("video"):
                                output_video = os.path.join("output", "test", os.path.basename(test_file_path))
                                if os.path.exists(output_video):
                                    st.video(output_video)
                                else:
                                    st.error("Failed to generate output video.")
                            
                            elif use_webcam:
                                st.write("Webcam functionality requires additional configuration and isn't supported in this demo.")
                else:
                    st.warning("Please upload a test file or enable webcam.")
        
        with col2:
            st.markdown('<div class="sub-header">Recent Detections</div>', unsafe_allow_html=True)
            
            # Display recent detections if available
            test_output_dir = os.path.join("output", "test")
            if os.path.exists(test_output_dir):
                output_files = [f for f in os.listdir(test_output_dir) 
                               if f.endswith((".jpg", ".jpeg", ".png")) and f != "test_file.jpg"]
                
                if output_files:
                    output_files.sort(key=lambda x: os.path.getmtime(os.path.join(test_output_dir, x)), reverse=True)
                    recent_outputs = output_files[:5]  # Show the 5 most recent
                    
                    for output_file in recent_outputs:
                        st.image(
                            os.path.join(test_output_dir, output_file),
                            caption=output_file,
                            use_column_width=True
                        )
                else:
                    st.info("No recent detections found.")
            else:
                st.info("No detection results yet.")
            
            # Export options
            st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
            
            export_format = st.selectbox(
                "Export Format:",
                ["ONNX", "TensorFlow SavedModel", "TensorFlow Lite", "PyTorch"]
            )
            
            if st.button("Export Model", use_container_width=True):
                st.info(f"Export to {export_format} would be implemented here.")
                # In a real implementation, this would call YOLO's export functionality

# Add a footer with information
st.markdown("""---""")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h3>AI Training Gym</h3>
        <p>A complete pipeline for custom object detection</p>
    </div>
    <div>
        <p>Built with Streamlit, YOLOv5, and OpenCV</p>
    </div>
</div>
""", unsafe_allow_html=True)