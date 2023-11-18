import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.2.2","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks api key"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime, timedelta
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import sys
    import pyarrow as pa

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    fg = fs.get_feature_group("wine", version=1)
    
    feature_view = fs.get_feature_view("wine", version=1)
    
    batch_data = feature_view.get_batch_data()
    
    quality = fg.read()["quality"]


    


    y_pred = model.predict(batch_data)


    dataset_api = project.get_dataset_api()    

    y_pred_converted = [yp.as_py() if isinstance(yp, pa.lib.Int32Scalar) else yp for yp in y_pred]
    quality_converted = [ql.as_py() if isinstance(ql, pa.lib.Int32Scalar) else ql for ql in quality]
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=['datetime'],
                                                description="Wine flower Prediction/Outcome Monitoring"
                                                )
    
    
    
    now = datetime.now()
    data = {
        'prediction': [y_pred_converted],
        'label': [quality_converted],
        'datetime': [now],
       }

    # Now:
    """
    prediction                                         label                                             datetime
    0  [6, 6, 8, 8, 6, 7, 8, 6, 8, 8, 8, 6, 6, 8, 6, ...  [6, 6, 8, 8, 6, 7, 8, 6, 8, 8, 8, 6, 6, 8, 6, ... 2023-11-18 21:21:23   
    """
    # But I want:
    """
        prediction  label   datetime
    0   6           6       2023-11-18 21:21:23
    1   6           6       2023-11-18 21:21:24
    ...
    """
    data = {
        'prediction': [y_pred_converted],
        'label': [quality_converted],
        'datetime': [now],
    }
    #print(data_list)
    monitor_df = pd.DataFrame(data)
    # Because I have datetime as a unique, I'll increment it for each item 


    monitor_df = convert_data(monitor_df)
    

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])
    #print(len(history_df))
    print(history_df.head())
    monitor_fg.insert(monitor_df)


    # Create a confusion matrix
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    cm = confusion_matrix(labels, predictions)
    # save to file
    df_cm = pd.DataFrame(cm)
    print(df_cm.head())
    fig = pyplot.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    pyplot.xlabel("Predicted")
    pyplot.ylabel("Actual")
    pyplot.title("Wine Quality Confusion Matrix")
    pyplot.savefig("confusion_matrix_wine.png")
    pyplot.close(fig)
    # upload to hopsworks
    # file 
    #dataset_api.upload("confusion_matrix_wine.png", "Resources/confusion_matrix_wine+.png", overwrite=True)
    now = now.strftime('%Y-%m-%d_%H_%M_%S')
    dataset_api.upload("confusion_matrix_wine.png", f"Resources/confusion_matrix_wine_{now}.png", overwrite=True)


def convert_data(data):
    # Unpack the input data
    import pandas as pd
    prediction = data['prediction']
    label = data['label']
    datetime = data['datetime']
    from datetime import datetime, timedelta

    # Create a list of dictionaries where each dictionary represents one entry
    now = datetime.now()
    data_list = [{'prediction': p, 'label': l, 
                  'datetime': ((now + timedelta(seconds=i))
                               .strftime('%Y-%m-%d %H:%M:%S'))
                               } 
                 for i, (p, l) in enumerate(zip(prediction[0], label[0]))]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    return df

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

