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
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()

    # batch data split in quality and the rest
    quality = batch_data['quality']
    batch_data_to_predict = batch_data.drop(columns=['quality'])

    
    y_pred = model.predict(batch_data_to_predict)

    dataset_api = project.get_dataset_api()    

    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine flower Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    #data = {
    #    'prediction': [flower],
    #    'label': [label],
    #    'datetime': [now],
    #   }
    data = {
        'prediction': y_pred,
        'label': quality,
        'datetime': now,
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})


    # Create a confusion matrix
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    print("Number of different wine labels to date: " + str(labels.value_counts().count()))

    # Create a confusion matrix
    cm = confusion_matrix(labels, predictions)
    #print(cm)
    # save to file
    df_cm = pd.DataFrame(cm)
    fig = pyplot.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    pyplot.xlabel("Predicted")
    pyplot.ylabel("Actual")
    pyplot.title("Wine Quality Confusion Matrix")
    pyplot.savefig("confusion_matrix.png")
    pyplot.close(fig)
    # upload to hopsworks
    dataset_api.upload("confusion_matrix.png", "Resources/confusion_matrix.png")



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

