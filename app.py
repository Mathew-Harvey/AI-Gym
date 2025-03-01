import streamlit as st
import cv2
import os
import json
import random
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Create necessary directories
for dir in ["dataset/images", "dataset/labels", "annotations", "models", "output"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames from a video at a specified frame rate."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps) if fps > 0 else 1
    while success:
        if count % interval == 0:
            frame_name = f"{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()

# Streamlit app configuration
st.title("AI Training Gym")
st.write("A tool to upload data, annotate images, train a YOLOv5 model, and detect objects.")
menu = ["Home", "Class Definition", "Annotation", "Training", "Inference"]
choice = st.sidebar.selectbox("Select Step", menu)

# Home Section: Upload images or videos
if choice == "Home":
    st.subheader("Upload Training Data")
    st.write("Upload images or videos to create your dataset. Videos will be converted to frames.")
    uploaded_files = st.file_uploader(
        "Upload images or videos", 
        accept_multiple_files=True, 
        type=["jpg", "png", "mp4"]
    )
    fps = st.number_input("Frame extraction rate (frames per second)", min_value=0.1, value=1.0)
    if st.button("Process"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("dataset/images", uploaded_file.name)
                if uploaded_file.type.startswith("image"):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                elif uploaded_file.type.startswith("video"):
                    temp_path = "temp_video.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    extract_frames(temp_path, "dataset/images", fps)
                    os.remove(temp_path)
            st.success("Data processed successfully!")
        else:
            st.warning("Please upload at least one file.")

# Class Definition Section: Define object classes
elif choice == "Class Definition":
    st.subheader("Define Classes")
    st.write("Enter the names of the objects you want to detect, one per line.")
    classes_text = st.text_area("Enter class names (one per line)", placeholder="cat\ndog\ncar")
    if st.button("Save Classes"):
        if classes_text.strip():
            with open("classes.txt", "w") as f:
                f.write(classes_text)
            st.success("Classes saved to 'classes.txt'.")
        else:
            st.error("Please enter at least one class name.")

# Annotation Section: Annotate images with bounding boxes
elif choice == "Annotation":
    st.subheader("Annotate Images")
    st.write("Select an image, draw bounding boxes, assign classes, and save annotations.")
    if not os.path.exists("classes.txt"):
        st.error("Please define classes first in the 'Class Definition' section.")
    else:
        with open("classes.txt") as f:
            classes = [line.strip() for line in f.readlines()]
        images = [f for f in os.listdir("dataset/images") if f.endswith((".jpg", ".png"))]
        if not images:
            st.warning("No images found. Please upload data in the 'Home' section.")
        else:
            img_select = st.sidebar.selectbox("Choose Image", images)
            img_path = os.path.join("dataset/images", img_select)
            img = Image.open(img_path)
            width, height = img.size

            # Load existing annotations
            annotation_file = os.path.join("annotations", img_select + ".json")
            if os.path.exists(annotation_file):
                with open(annotation_file) as f:
                    annotation_data = json.load(f)
                    st.session_state.annotations = annotation_data["boxes"]
            else:
                st.session_state.annotations = []

            # Display canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
                stroke_width=2,
                background_image=img,
                update_streamlit=True,
                height=height,
                width=width,
                drawing_mode="rect",
                key=f"canvas_{img_select}",
            )

            # Add new bounding box
            class_label = st.selectbox("Select Class", classes)
            if st.button("Add Box"):
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        last_object = objects[-1]
                        if last_object["type"] == "rect":
                            box = {
                                "class": class_label,
                                "left": last_object["left"],
                                "top": last_object["top"],
                                "width": last_object["width"],
                                "height": last_object["height"]
                            }
                            st.session_state.annotations.append(box)
                            st.success(f"Added box for '{class_label}'.")

            # Display and manage current annotations
            if st.session_state.annotations:
                st.write("Current Annotations:")
                for i, box in enumerate(st.session_state.annotations):
                    st.write(
                        f"Box {i+1}: {box['class']} at ({box['left']:.1f}, {box['top']:.1f}), "
                        f"size ({box['width']:.1f}, {box['height']:.1f})"
                    )
                    if st.button(f"Delete Box {i+1}", key=f"del_{img_select}_{i}"):
                        st.session_state.annotations.pop(i)
                        st.experimental_rerun()

            # Save annotations
            if st.button("Save Annotations"):
                annotation_data = {
                    "image": img_select,
                    "width": width,
                    "height": height,
                    "boxes": st.session_state.annotations
                }
                os.makedirs("annotations", exist_ok=True)
                with open(annotation_file, "w") as f:
                    json.dump(annotation_data, f)
                st.success("Annotations saved successfully.")

# Training Section: Train the YOLOv5 model
elif choice == "Training":
    st.subheader("Train Your Model")
    st.write("Train a YOLOv5 model using your annotated dataset. This may take some time.")
    model_name = st.text_input("Model Name", value="custom_model")
    if st.button("Start Training"):
        if not os.path.exists("classes.txt"):
            st.error("Please define classes first.")
        elif not os.listdir("dataset/images"):
            st.error("Please upload images first.")
        else:
            # Convert annotations to YOLO format
            with open("classes.txt") as f:
                classes = [line.strip() for line in f.readlines()]
            for img in os.listdir("dataset/images"):
                if img.endswith((".jpg", ".png")):
                    annotation_file = os.path.join("annotations", img + ".json")
                    label_file = os.path.join("dataset/labels", img.replace(".jpg", ".txt").replace(".png", ".txt"))
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

            # Split dataset into train and validation sets
            all_images = [os.path.join("dataset/images", f) for f in os.listdir("dataset/images") if f.endswith((".jpg", ".png"))]
            random.shuffle(all_images)
            split_index = int(0.8 * len(all_images)) if len(all_images) > 1 else len(all_images)
            train_images = all_images[:split_index]
            val_images = all_images[split_index:] if split_index < len(all_images) else train_images

            with open("dataset/train.txt", "w") as f:
                for img in train_images:
                    f.write(f"{img}\n")
            with open("dataset/val.txt", "w") as f:
                for img in val_images:
                    f.write(f"{img}\n")

            # Create dataset.yaml
            with open("dataset.yaml", "w") as f:
                f.write("train: dataset/train.txt\n")
                f.write("val: dataset/val.txt\n")
                f.write(f"nc: {len(classes)}\n")
                f.write(f"names: {classes}\n")

            # Train the model
            st.info("Training started. This may take a while...")
            model = YOLO('yolov5s.pt')  # Start with small YOLOv5 model
            model.train(data='dataset.yaml', epochs=50, project='models', name=model_name)
            st.success(f"Training completed! Model saved in 'models/{model_name}'.")

# Inference Section: Test the trained model
elif choice == "Inference":
    st.subheader("Test Your Model")
    st.write("Upload an image or video to detect objects using your trained model.")
    model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
    if not model_dirs:
        st.warning("No trained models found. Please train a model first.")
    else:
        model_select = st.selectbox("Select Model", model_dirs)
        test_file = st.file_uploader("Upload Test File", type=["jpg", "png", "mp4"])
        if st.button("Detect"):
            if test_file is not None:
                # Save test file with original extension
                test_extension = os.path.splitext(test_file.name)[1]
                test_file_path = os.path.join("output", f"test_file{test_extension}")
                with open(test_file_path, "wb") as f:
                    f.write(test_file.getbuffer())

                # Load model and perform inference
                model_path = os.path.join("models", model_select, "weights", "best.pt")
                model = YOLO(model_path)
                results = model(test_file_path, save=True, project="output", name="test")

                # Display results
                if test_file.type.startswith("image"):
                    output_img = os.path.join("output", "test", f"test_file.jpg")
                    if os.path.exists(output_img):
                        st.image(output_img, caption="Detection Result")
                    else:
                        st.error("Failed to generate output image.")
                elif test_file.type.startswith("video"):
                    output_video = os.path.join("output", "test", f"test_file.mp4")
                    if os.path.exists(output_video):
                        st.video(output_video)
                    else:
                        st.error("Failed to generate output video.")
            else:
                st.warning("Please upload a test file.")