# ID 2223 Scalable Machine Learning and Deep Learning

## Assignment - Train Whisper Model


### Part 1 - Downloading the data and saving it to Google Drive

This Part was completed in the following [Jupyter Notebook](/Lab2/Lab2-Part1(Download).ipynb).

### Part 2 - Preprocessing the data

This Part was completed in the following [Jupyter Notebook](/Lab2/Lab2-Part2.ipynb).

### Part 3 - Training the model

This Part was completed in the following [Jupyter Notebook](/Lab2/Lab2-Part3.ipynb).

### Part 4 - Deploying the model

This Part was completed in the following [Python script1](/Lab2/UI/app.py) and [Python script2](/Lab2/UI/llama.py).

# Explanation:

### Chunking the input data.

The unprocessed input data for the training is a total of 22GB. 
This ammounts to a total of 496,000 input rows of training data and 16,000 input rows of validation data.
Since this data is way too large to keep it in ram or stored on disk as a single file, we need to chunk it into smaller pieces.
We chunked the input data into chunks of 1000 rows for a total of 496 chunks of training data and 16 chunks of validation data.
This allows us to keep the data in ram and process it in batches.

### Training the model.

The Training of the model is done in the following steps:
1. We read the data from the arrow files in chunks of 1000 rows.
2. We then train with the data on the current model.
3. We then evaluate the model with a random subset of 250 rows of the validation data.
4. We then save the better performing model to disk.

This process however has a few flaws:

1. Since we are training on a subset of the data, we are not training on the full dataset. This however would simply not be feasible with the ammount of data we have.
2. Since we are evaluating the model on a subset of the validation data, we are not getting a good representation of the models performance. Our assumption was that the model will still however over many iterations converge to a good model. This assumption was proven to be incorrect as the final model did perform similarly to the original model.


### Something about the UI (!!!WIP!!!)



### Evaluation for the model:

The evaluation was done once the training was completed for a sizeable amount of the input chunks. 
This time the evaluation was done on the whole validation dataset.

These runs took about 2 hours and were done on a local machine with a 4060ti GPU.

The results of the evaluation can be seen in the following table:

| Model | WER |
| --- | --- |
| Whisper-tiny                          | 46.68657  |
| Whisper-small-trained-20k-rows        | 18.49936  |
| Whisper-small                         | 18.48644  |
| Whisper-medium (eval time: 4,5h)      | 12.34470  |



### Explanation for the for the evaluation:

According to OpenAI, about a third of Whisper's audio dataset is non-English, and it is given the task of transcribing or translating to English. 
According to their [paper about the whisper dataset](https://cdn.openai.com/papers/whisper.pdf) 13344 hours of german audio data was used for training. 
The model performs reasonably well with German, which is a significant portion of the dataset. 
However, if you already have a well-performing model trained on a specific language, the improvement in performance for other languages may be minimal. 
This is because the Whisper model has not been specifically trained on those languages. It is recommended to consider using a larger or better base model, as the performance difference between the tiny and small models suggests.

Our assumption is that the training would have needed a lot more time to converge to a better model.
The problem is that our 1k runs took about 2 hours to complete.
Our assumption is that the model would have needed about 100 rounds of training to converge to a better model.
This would have taken about 200 hours to complete.
This is why we decided to stop the training after 20 runs.

### Conclusion:

https://cdn.openai.com/papers/whisper.pdf
https://github.com/openai/whisper