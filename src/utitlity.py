import os
from transformers import BertTokenizer, BertForSequenceClassification,pipeline,AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pandas as pd
import csv
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from summarizer import Summarizer
import re

class Utility():

    def __init__(self):
        self.sentence = None
        self.kpi = {
        "Waste_And_Hazardous_Materials_Management": "Toxic_Emissions_And_Waste",
        "Water_And_Wastewater_Management": "Supply_Chain_Management",
        "Supply_Chain_Management":"Supply_Chain_Management",
        "Systemic_Risk_Management":"Customer_Others",
        "Access_And_Affordability": "Others",
        "Air_Quality": "Toxic_Emissions_And_Waste",
        "Business_Ethics": "Customer_Relations",
        "Business_Model_Resilience": "Customer_Others",
        "Competitive_Behavior": "AntiCompetetive_Prcatices",
        "Critical_Incident_Risk_Management": "Others",
        "Customer_Privacy": "Privacy_And_Data_Security",
        "Customer_Welfare": "Customer_Relations",
        "Data_Security": "Privacy_And_Data_Security",
        "Director_Removal": "Social_Others",
        "Ecological_Impacts": "Envionmental_Others",
        "Employee_Engagement_Inclusion_And_Diversity": "Discimination_And_Worforce_Diversity",
        "Employee_Health_And_Safety": "Helath_And_Safety",
        "Energy_Management": "Environmental_Others",
        "GHG_Emissions": "Toxic_Emissions_And_Waste",
        "Human_Rights_And_Community_Relations": "Customer_Relations",
        "Labor_Practices": "Supply_And_Labor_Chain_Standards",
        "Management_Of_Legal_And_Regulatory_Framework": "Social_Others",
        "Physical_Impacts_Of_Climate_Change": "Energy_And_Climate_Change",
        "Product_Design_And_Lifecycle_Management": "Others",
        "Product_Quality_And_Safety": "Others",
        "Selling_Practices_And_Product_Labeling": "Others",
    }
    
    def prediction_classifier(self,sentence):
        finBert=BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
        tokeniZer=BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
        esgScore=pipeline("text-classification",model=finBert,tokenizer=tokeniZer)
        try:
            score = esgScore(sentence)
        except Exception as e:
            score = None
        return score

    def text_classifier(self,sentence):
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        classifier=pipeline(task="text-classification",model=model,tokenizer=tokenizer)
        try:
            score = classifier(sentence)
        except Exception as e:
            score = None
        return score

    def clean_text(self,text):
        # Remove non ASCII characters
        # Replace tabs with spaces
        line = re.sub(r"\t+", r" ", text)
        url_str = (r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\."
                r"([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
        line = re.sub(url_str, r" ", line)  # URLs
        line = re.sub(r"^\s?\d+(.*)$", r"\1", line)  # headers
        line = re.sub(r"\d{5,}", r" ", text)  # figures
        line = re.sub(r"\.+", ".", text)  # multiple periods

        line = line.strip()  # leading & trailing spaces
        line = re.sub(r"\s+", " ", line)  # multiple spaces
        line = re.sub(r"\s?([,:;\.])", r"\1", line)  # punctuation spaces
        line = re.sub(r"\s?-\s?", "-", line)  # split-line words
        # Aggregate lines where the sentence wraps
        # Also, lines in CAPITALS is counted as a header

        # Clean the lines into sentences
        return line

    def summary_text(self,text):
        model = Summarizer()
        result = model(text, min_length=60,)
        full = ''.join(result)
        return full

    def process_sub_pillars(self,sentence):
        tokenizer = AutoTokenizer.from_pretrained("nbroad/ESG-BERT")
        esgbert = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")
        nlp = pipeline("text-classification", model=esgbert, tokenizer=tokenizer)
        kpi = []
        try:
            score = nlp(sentence,return_all_scores=True)
            if score:
                for i in score[0]:
                    if i['score'] > 0.01:
                        kpi.append(self.kpi[i['label']])
        except Exception as e:
            kpi = None
        return kpi


    def process_org(self, sentence):
        exclude = set(string.punctuation)
        st = ''.join(ch for ch in sentence if ch not in exclude)
        nlp = en_core_web_sm.load()
        doc = nlp(st)
        org_cnt = {}
        for X in doc.ents:
            if X.label_ == 'ORG' :
                if X.text in org_cnt:
                    org_cnt[X.text] = org_cnt[X.text] + 1
                else:
                    org_cnt[X.text] = 1
        return org_cnt
