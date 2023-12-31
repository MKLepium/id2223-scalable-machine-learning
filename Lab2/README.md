# ID 2223 Scalable Machine Learning and Deep Learning

Link to huggingface spaces: https://huggingface.co/spaces/willeasp/voice-chat-german

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
Each of those preprocessed chunks is about 1 GB in size.
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


### Service - Voice Chat in German

The goal of the service is to provide a voice chat in german, using the whisper model for speech recognition, and the llama model for generated text for the chat function. 
The google text-to-speech API is used to generate the audio that is read back to the user. 

The interface is built using gradio blocks, and provides microphone and text input options.
The chat is presented using the gradio Chatbot compoenent. 

We found that we had a hard time with llama, since it often only returns a zero-width space character (`"\u200b"`), which is not visible in the chat.
However, this is not a problem with the service, but rather with the llama model itself.

The service can be found at https://huggingface.co/spaces/willeasp/voice-chat-german


### Evaluation for the model:

The evaluation was done once the training was completed for a sizeable amount of the input chunks. 
This time the evaluation was done on the whole validation dataset.

These runs took about 2 hours and were done on a local machine with a 4060ti GPU.

The results of the evaluation can be seen in the following table:

| Model | WER |
| --- | --- |
| Whisper-tiny                                              | 46.68657  |
| Whisper-small-trained-20k-rows                            | 18.49936  |
| Whisper-small                                             | 18.48644  |
| Whisper-medium (eval time: 252,7 Minutes)                 | 12.34470  |
| Whisper-medium on Dardel (eval time: 66,1 Minutes)        | 12.34674  |

We also had to run the Whisper-medium multiple times on google colab, since the local was not feasible for the medium model. The VRam of the 4060ti was just not enough.
Additionally we had to run it multiple times since we ran out of computation time on colab a few times.



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

### Conclusions:

Our training did not worsen or improve the model. 
Our assumption is that we would have needed to train the model for a lot longer to see significant improvements.
We also think that since the german language is a significant portion of the dataset, the base-model performs reasonably well on german.
This is why we think that the model would have needed a lot more time to converge to a better model.


https://cdn.openai.com/papers/whisper.pdf
https://github.com/openai/whisper