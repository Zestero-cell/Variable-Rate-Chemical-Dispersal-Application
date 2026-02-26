import streamlit as st
import time
import os
import re
import json
import folium
from streamlit_folium import st_folium
from PIL import Image

# --- CONFIGURATION ---
INTERMEDIATE_DIR = "Intermediate_Images"
OUTPUT_IMAGE_DIR = "Output_Images"
VRA_OUTPUT_DIR = "VRA_Output"
SPRAYPOINTS_DIR = "Spraypoints"

st.set_page_config(page_title="Agricultural Weed Detection System", layout="wide")

st.title("Topological Separation Of Occluded Vegetation In Banana Plantations Using Prototype-Guided Graph Attention Networks")
st.markdown("---")

# Initialize session state for tracking displayed phases
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

        # Button to start simulation
        if st.button("Start Fast Test (10 Seconds)"):
            # Reset tracking flags for a fresh run
            st.session_state.phases_shown = {k: False for k in st.session_state.phases_shown}
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Placeholders to prevent duplicate printing
            placeholder_inter = st.empty()
            placeholder_pred = st.empty()
            placeholder_geo = st.empty()
            placeholder_vra = st.empty()

            # TESTING CONFIGURATION: 10 Seconds total
            total_seconds = 10 
            start_time = time.time()
            
            # File Paths
            inter_path = os.path.join(INTERMEDIATE_DIR, f"DJI_{unique_id}.JPG")
            pred_path = os.path.join(OUTPUT_IMAGE_DIR, f"DJI_{unique_id}_pred.png")
            geojson_path = os.path.join(SPRAYPOINTS_DIR, f"3_weed_map_{unique_id}.geojson")
            vra_path = os.path.join(VRA_OUTPUT_DIR, f"graph_{unique_id}.png")

            while True:
                elapsed = time.time() - start_time
                percent = min(int((elapsed / total_seconds) * 100), 100)
                progress_bar.progress(percent)

                # --- PHASE 1: 2 Seconds (Intermediate) ---
                if elapsed >= 2 and not st.session_state.phases_shown["intermediate"]:
                    with placeholder_inter.container():
                        st.markdown("---")
                        st.header("Step 1: Intermediate Pre-processing")
                        if os.path.exists(inter_path):
                            st.image(inter_path, caption="Feature Extraction / SLIC Segmentation", width=700)
                        else:
                            st.error(f"File not found: {inter_path}")
                    st.session_state.phases_shown["intermediate"] = True

                # --- PHASE 2: 5 Seconds (Output Prediction) ---
                if elapsed >= 5 and not st.session_state.phases_shown["prediction"]:
                    with placeholder_pred.container():
                        st.markdown("---")
                        st.header("Step 2: Occlusion-Aware Prediction")
                        if os.path.exists(pred_path):
                            st.image(pred_path, caption="YOLOv8 + ResNet-50 Detection Mask", width=700)
                        else:
                            st.error(f"File not found: {pred_path}")
                    st.session_state.phases_shown["prediction"] = True

                # --- PHASE 3: 7 Seconds (GeoJSON & Map) ---
                if elapsed >= 7 and not st.session_state.phases_shown["geojson"]:
                    with placeholder_geo.container():
                        st.markdown("---")
                        st.header("Step 3: Geospatial Mapping")
                        if os.path.exists(geojson_path):
                            with open(geojson_path, "r") as f:
                                geo_data = json.load(f)
                            try:
                                coords = geo_data['features'][0]['geometry']['coordinates']
                                m = folium.Map(location=[coords[1], coords[0]], zoom_start=19)
                                folium.GeoJson(geo_data, marker=folium.CircleMarker(radius=3, color="red", fill=True)).add_to(m)
                                st_folium(m, width=1000, height=400, key=f"map_{unique_id}")
                            except:
                                st.warning("Map failed to load.")
                            st.download_button(label="📥 Download GeoJSON", data=json.dumps(geo_data), file_name=f"weed_map_{unique_id}.geojson", mime="application/json")
                    st.session_state.phases_shown["geojson"] = True

                # --- PHASE 4: 10 Seconds (VRA Output) ---
                if elapsed >= 10:
                    if not st.session_state.phases_shown["vra"]:
                        with placeholder_vra.container():
                            st.markdown("---")
                            st.header("Step 4: VRA Graph Analysis")
                            if os.path.exists(vra_path):
                                st.image(vra_path, caption="Final Graph Attention Output", width=700)
                        st.session_state.phases_shown["vra"] = True
                    
                    status_text.success("✅ Test Pipeline Completed!")
                    st.balloons()
                    break

                # Dynamic Status Messages
                status_text.info(f"⏳ Running Fast Test... {int(elapsed)}s / 10s")
                time.sleep(0.5) # Fast refresh for testing

    else:
        st.error("Filename format incorrect. Use DJI_$$$$.JPG")
else:
    st.info("Upload a drone image to begin the 10-second test.")
