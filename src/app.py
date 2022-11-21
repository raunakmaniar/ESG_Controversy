from flask import Flask, render_template, request,json,jsonify
# from flask_cors import CORS,cross_origin
import os 
import re
from utitlity import Utility
import csv
app = Flask(__name__, static_url_path='/static')
# cors = CORS(app,resources={r"/":{"origins":"*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/', endpoint="upload_file")
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'], endpoint="uploader")
# @cross_origin(origins='', headers=['Content-Type', 'Authorization'])
def uploader():
    if request.method == 'POST':

        utillity_obj =  Utility()
        file_list = []
        esg_chart = {'Environmental':0,'Social':0,'Goverance':0,'None':0}
        dir = os.path.join(os.getcwd(),"dataset2")
        for file in os.listdir(dir):
            file_path = os.path.join(dir,file)
            try:
                with open(file_path,'r',encoding="utf-8") as f:
                    sentence = f.read()
                    sentence = utillity_obj.summary_text(sentence)
                    print('Sentence',sentence)
                    score_esg=utillity_obj.prediction_classifier(sentence)
                    if score_esg != None:
                        label = score_esg[0]['label']
                        if score_esg[0]['label'] != None:
                            org = utillity_obj.process_org(sentence)
                            rating = utillity_obj.text_classifier(sentence)
                            sub_pillar = utillity_obj.process_sub_pillars(sentence)
                            file_list.append({'Filename':file,'Direct org':org, 'Sub pillars':sub_pillar ,'Nature of harm':'','Scale of impact':rating, 'Controversy score':score_esg,'Ongoing':''})    
                        else:
                            file_list.append({'Filename':file,'Direct org':'', 'Sub pillars':'' ,'Nature of harm':'','Scale of impact':rating, 'Controversy score':score_esg,'Ongoing':''})  
                        esg_chart[label] = esg_chart[label] + 1
            except Exception as e:
                print("Exception",e)
                return {'Data':file_list,'chart':esg_chart}
        with open('output.csv', 'w', newline='') as file:
            fieldnames = ['Filename','Direct org', 'Sub pillars' ,'Nature of harm','Scale of impact', 'Controversy score','Ongoing']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            for res in file_list:
                if res != None:
                    writer.writerow(res)
        esg_chart = {'Environmental': 2,'Goverance': 0,'None': 1,'Social': 8}
        return {'Data':file_list,'chart':esg_chart}

# 
if __name__ == '__main__':
    app.run(debug=True)
