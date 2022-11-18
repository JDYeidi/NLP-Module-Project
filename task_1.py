#Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class movies_reviews():
    def __init__(self):
        #Preparing model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.classifier = pipeline("sentiment-analysis", model = self.model, tokenizer = self.tokenizer)
        self.dataset = 'tiny_movie_reviews_dataset.txt'

    def main(self):
        with open(self.dataset, 'r') as f:
            reviews = f.readlines()

        for i, review in enumerate(reviews): 
            inputs = review
            output = self.classifier(inputs, max_length=512, truncation=True)
            if output[0]['label'] == 'LABEL_0':
                print("Review " + str(i) + ": " + "Negative")
            elif output[0]['label'] == 'LABEL_1':
                print("Review " + str(i) + ": " + "Neutral")
            elif output[0]['label'] == 'LABEL_2':
                print("Review " + str(i) + ": " + "Positive")
    
