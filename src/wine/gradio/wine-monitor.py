import gradio as gr
from PIL import Image
import hopsworks
import os

# add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import Constants


project = hopsworks.login(project="ID2223_MKLepium")
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download(os.path.join(Constants.HOPS_IMAGE_PATH, Constants.DF_RECENT_NAME), overwrite=True)
dataset_api.download(os.path.join(Constants.HOPS_IMAGE_PATH, Constants.CONFUSION_MATRIX_NAME), overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image(Constants.DF_RECENT_NAME, elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image(Constants.CONFUSION_MATRIX_NAME, elem_id="confusion-matrix")        

demo.launch(server_port=8081, server_name="0.0.0.0")
