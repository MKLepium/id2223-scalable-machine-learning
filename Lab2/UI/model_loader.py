import os
from transformers import WhisperForConditionalGeneration

class model_loader():
    def __init__(self, path = "/home/mk/trained_model/snapshot_step_27"):
        self.path = path
        self.model = WhisperForConditionalGeneration.from_pretrained(self.path)

