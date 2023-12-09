# ID 2223 Scalable Machine Learning and Deep Learning

# For Lab 2 see the [Lab2 folder](/Lab2)

## Assignment 1

### Task 1 - Iris classification

You will find the code for this task in the src/iris folder. 
The code is split into 2 jupyter notebooks and 4 python files.
The training pipeline and the exploratory data analysis are in the notebooks. 
The batch-inference pipeline aswell as the feature pipeline are in the python files.
The final two files are in the gradio folder that replace the huggingface individual apps with two web apps.

The two files that run daily are the batch-inference pipeline and the feature pipeline:
These are ran using the workflows defined in .github/workflows/ "iris-batch-inference-daily.yml" and "iris-feature-pipeline-daily.yml".
We delay the batch inference pipeline by 1 hour to ensure that the feature pipeline has finished running.

URL for iris-tester:
[Selfhosted](http://88.99.215.78:8082/)
Screenshot gradio app for the iris-tester:
![Screenshot gradio app for the iris-tester](/screenshots/Iris-Tester.PNG)

Screenshot gradio app for the iris-monitor:
[Selfhosted](http://88.99.215.78:8081/)
![Screenshot gradio app for the iris-monitor](/screenshots/Iris-Monitor.PNG)

### Task 2 - Wine quality 

You will find the code for this task in the src/wine folder.
The code is split into 2 jupyter notebooks and 5 python files.
The training pipeline and the exploratory data analysis are in the notebooks.
The batch-inference pipeline aswell as the feature pipeline are in the python files. 
Additionally we have a "find_min_max_values_for_wine.py" file that is used to help us generate the sample data for the feature pipeline.
The final two files are in the gradio folder that replace the huggingface individual apps with two web apps.

URL for Wine-tester:
[Selfhosted](http://88.99.215.78:8084/)
Screenshot gradio app for the Wine-tester:
![Screenshot gradio app for the wine-tester](/screenshots/Wine-Tester.PNG)

Screenshot gradio app for the Wine-monitor:
[Selfhosted](http://88.99.215.78:8083/)
![Screenshot gradio app for the wine-monitor](/screenshots/Wine-Monitor.PNG)

