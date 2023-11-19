import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine_prediction(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol, is_red):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    # we currently ignore citric_acid since it is not part of the training model
    # It performed better without
    if is_red:
        red= 1
    else:
        red= -1
    
    df = [
    fixed_acidity, volatile_acidity, 
    residual_sugar, chlorides, 
    free_sulfur_dioxide, total_sulfur_dioxide, 
    density, ph, sulphates, alcohol, red
    ],

    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)    
    return res[0]
        
# input:	fixed_acidity	volatile_acidity	citric_acid	residual_sugar	chlorides	free_sulfur_dioxide	total_sulfur_dioxide	density	ph	sulphates	alcohol

demo = gr.Interface(
    fn=wine_prediction,
    title="Wine  Predictive Analytics",
    description="Experiment with wine values to predict quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default = 5.672923867348613, label="fixed_acidity"),
        gr.inputs.Number(default = 0.3119492194858408, label="volatile_acidity"),
        gr.inputs.Number(default = 0.37579835181635224, label="citric_acid"),
        gr.inputs.Number(default = 0.0, label="residual_sugar"),
        gr.inputs.Number(default = 0.023688360714891565, label="chlorides"),
        gr.inputs.Number(default = 16.23835123754711, label="free_sulfur_dioxide"),
        gr.inputs.Number(default = 266.1489881380038, label="total_sulfur_dioxide"),
        gr.inputs.Number(default = 0.9914648779574159, label="density"),
        gr.inputs.Number(default = 3.130146817441741, label="ph"),
        gr.inputs.Number(default = 0.6314288256824422, label="sulphates"),
        gr.inputs.Number(default = 7.777006000294087, label="alcohol"),
        gr.inputs.Radio(choices=["red_whine", "white_wine"], label="is_red"),
        ],
    # outputs should be text
    outputs=gr.Textbox(type="text")
    )

demo.launch(debug=True, server_port=8084, server_name="0.0.0.0")

