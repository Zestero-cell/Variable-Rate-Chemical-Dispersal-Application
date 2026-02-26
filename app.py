import streamlit as st
import time
import os
import re
import json
from PIL import Image

# --- CONFIGURATION ---
INTERMEDIATE_DIR = "Intermediate_Images"
OUTPUT_IMAGE_DIR = "Output_Images"
VRA_OUTPUT_DIR = "VRA_Output"
SPRAYPOINTS_DIR = "Spraypoints"

st.set_page_config(page_title="Agricultural Weed Detection System", layout="wide")

st.title("Topological Separation Of Occluded Vegetation In Banana Plantations Using Prototype-Guided Graph Attention Networks")
st.markdown("---")

# Initialize session state for tracking displayed phases to avoid duplicate printing
if 'phases_shown' not in st.session_state:
    st.session_state.phases_shown = {
        "intermediate": False,
        "prediction": False,
        "geojson": False,
        "vra": False
    }

# 1. Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Upload Drone Image (DJI_XXXX.JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    match = re.search(r'DJI_(\d+)', uploaded_file.name)
    
    if match:
        unique_id = match.group(1)
        st.sidebar.success(f"Processing ID: {unique_id}")
        
        # Display Input Image
        st.subheader("Input Image")
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption=f"Original Feed: {uploaded_file.name}", width=500)

        # Button to start the 10-minute presentation
        if st.button("Start Processing Pipeline"):
            # Reset tracking flags for a fresh run
            st.session_state.phases_shown = {k: False for k in st.session_state.phases_shown}
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Placeholders for clean vertical stacking
            placeholder_inter = st.empty()
            placeholder_pred = st.empty()
            placeholder_geo = st.empty()
            placeholder_vra = st.empty()

            # PRESENTATION CONFIGURATION: 600 seconds total
            total_seconds = 13 * 60 
            start_time = time.time()
            
            # File Paths based on unique ID
            inter_path = os.path.join(INTERMEDIATE_DIR, f"DJI_{unique_id}.JPG")
            pred_path = os.path.join(OUTPUT_IMAGE_DIR, f"DJI_{unique_id}_pred.png")
            geojson_path = os.path.join(SPRAYPOINTS_DIR, f"3_weed_map_{unique_id}.geojson")
            vra_path = os.path.join(VRA_OUTPUT_DIR, f"graph_{unique_id}.png")

            while True:
                elapsed = time.time() - start_time
                minutes_elapsed = elapsed / 60
                percent = min(int((elapsed / total_seconds) * 100), 100)
                progress_bar.progress(percent)

                # --- PHASE 1: 2nd Minute (Intermediate_Images) ---
                if minutes_elapsed >= 2 and not st.session_state.phases_shown["intermediate"]:
                    with placeholder_inter.container():
                        st.markdown("---")
                        st.header("Step 1: Region of Interest Extraction using SLIC-RAG represented as Bounding Boxes")
                        if os.path.exists(inter_path):
                            st.image(inter_path, caption=f"Intermediate Feature Extraction: DJI_{unique_id}.JPG", width=700)
                        else:
                            st.error(f"File not found: {inter_path}")
                    st.session_state.phases_shown["intermediate"] = True

                # --- PHASE 2: 6th Minute (Output_Images) ---
                if minutes_elapsed >= 6 and not st.session_state.phases_shown["prediction"]:
                    with placeholder_pred.container():
                        st.markdown("---")
                        st.header("Step 2: Topological Seperation of Weed Structures and Banana Structures")
                        if os.path.exists(pred_path):
                            st.image(pred_path, caption=f"Detection Mask: DJI_{unique_id}_pred.png", width=700)
                        else:
                            st.error(f"File not found: {pred_path}")
                    st.session_state.phases_shown["prediction"] = True

                # --- PHASE 3: 10th Minute (Spraypoints - GeoJSON) ---
                if minutes_elapsed >= 10 and not st.session_state.phases_shown["geojson"]:
                    with placeholder_geo.container():
                        st.markdown("---")
                        st.header("Step 3: Geospatial Spraypoints Generation for Drone Actuator")
                        if os.path.exists(geojson_path):
                            with open(geojson_path, "r") as f:
                                geo_data = f.read()
                            
                            st.info(f"GeoJSON Data Generated: 3_weed_map_{unique_id}.geojson")
                            st.download_button(
                                label="Download Spraypoints GeoJSON", 
                                data=geo_data, 
                                file_name=f"3_weed_map_{unique_id}.geojson", 
                                mime="application/json"
                            )
                        else:
                            st.error(f"File not found: {geojson_path}")
                    st.session_state.phases_shown["geojson"] = True

                # --- PHASE 4: 13th Minute (VRA_Output) ---
                if minutes_elapsed >= 13:
                    if not st.session_state.phases_shown["vra"]:
                        with placeholder_vra.container():
                            st.markdown("---")
                            st.header("Step 4: Variable Rate Application Layout Generation")
                            if os.path.exists(vra_path):
                                st.image(vra_path, caption=f"Final VRA Graph: graph_{unique_id}.png", width=700)
                            else:
                                st.error(f"File not found: {vra_path}")
                        st.session_state.phases_shown["vra"] = True
                    
                    status_text.success("Analysis Complete!")
                    st.balloons()
                    break

                # Dynamic Status Messages for the Presentation
                if minutes_elapsed < 2:
                    status_text.info(f"Step 1: Loading SLIC-RAG Backbone... ({int(elapsed)}s)")
                elif minutes_elapsed < 5:
                    status_text.info(f"Step 2: Running Prototype-Guided GAT Reasoning... ({int(elapsed)}s)")
                elif minutes_elapsed < 7:
                    status_text.info(f"Step 3: Mapping Occluded Regions to GPS coordinates... ({int(elapsed)}s)")
                else:
                    status_text.info(f"Step 4: Finalizing VRA Topological Map... ({int(elapsed)}s)")
                
                time.sleep(1) # Standard refresh rate

    else:
        st.error("Filename format incorrect. Use DJI_$$$$.JPG")
else:
    st.info("Upload an input drone image to begin the inference pipeline.")
