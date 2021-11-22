# Importing needed libraries
import numpy as np
import cv2
import time
from time import gmtime, strftime 
import argparse #เม้าส์ ROI
from tkinter import * #popup GUI ตั้งชื่อ ROI
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from folium.plugins import MarkerCluster
from folium.plugins import Search
from folium import plugins
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.features import DivIcon
import json
import pyrebase
import folium
import pandas
import requests
import xlsxwriter
import csv
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred)
db = firestore.client()


camera = cv2.VideoCapture(0)


h, w = None, None


with open(r"C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\modellnw\obj.names", 'r') as f:
 
    labels = [line.strip() for line in f]

num_of_class = np.zeros(len(labels), dtype=int) 


network = cv2.dnn.readNetFromDarknet(r"C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\modellnw\yolov3.cfg",
                                     r"C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\modellnw\modelverygood.weights")


layers_names_all = network.getLayerNames()


layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


probability_minimum = 0.5


threshold = 0.3


colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')



"""
End of:
Loading YOLO v3 network
"""

def detect():
        
        
        pic_ref = db.collection('countpic')
        docs = pic_ref.stream()
        d = list(docs)
        num = d[0].to_dict()["num"]
        c = num
        config = {
                                 
                                }
        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()
        global h,w
      
        while True:
            # Capturing frame-by-frame from camera
            _, frame = camera.read()
            # cv2.rectangle(frame,(0,186),(641,447),(0,0,255),5)
            if w is None or h is None:
        
                h, w = frame.shape[:2]


            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
            network.setInput(blob)  
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()
            bounding_boxes = []
            confidences = []
            class_numbers = []
            for result in output_from_network:
                
                for detected_objects in result:
                   
                    scores = detected_objects[5:]
                    
                    class_current = np.argmax(scores)
                
                    confidence_current = scores[class_current]

                  
                    if confidence_current > probability_minimum:
                        
                        box_current = detected_objects[0:4] * np.array([w, h, w, h])
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))
                        x_max = x_min + int(box_width)
                        y_max = y_min + int(box_height)
                                                                        
                        
                        if(x_min > 0 and y_min > 186 and x_max < 641 and y_max < 447):

                                print(labels[class_current])
                               
                                now = datetime.now()
                                date_time = now.strftime("%m/%d/%Y %I:%M:%S %p")
                                print(date_time)
                                
                                cv2.imwrite(r'C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\images'+str(c)+'.jpg',frame)
                                image_path = r'C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\images'+str(c)+'.jpg'
                                pic55 = cv2.imread(image_path)
                                pic55 = cv2.rectangle(pic55, (x_min,y_min), (x_max,y_max), (0,0,255), 1)
                                cv2.imwrite(r'C:\Users\Banana\Desktop\YOLO3OpenCV\YOLO-3-OpenCV\images'+str(c)+'.jpg',pic55)
                                path_on_cloud = "imagesnew"+ str(c) +".jpg"
                                path_local = "images"+ str(c) +".jpg"
                                storage.child(path_on_cloud).put(path_local)
                                t = storage.child(path_on_cloud).get_url(None)
                                num_of_class[class_current] = num_of_class[class_current]+1
                                location = db.collection('locations')
                                car_ref = db.collection('car')

                                doc_ref = db.collection('car').document()
                                doc_ref.set({
                                'class': labels[class_current],
                                'time': date_time,
                                'pic': "images"+ str(c) + ".jpg",
                                'path': t
                                })
                                c += 1
                                pic_ref = db.collection('countpic').document('count')
                                pic_ref.update({
                                'num': c
                                 })
                                
                                docs = car_ref.stream()
                               
                                time2 = date_time
                                

                                # if location.where('time','==',time2).stream():
                                #     doc2 = location.where('time','==',time2).stream()
                                #     d = list(doc2)
                                #     if d != []:
    
                                #         print(d[0].to_dict()["time"])
                                #         doc_refcol = db.collection('map2').document()
                                #         doc_refcol.set({
                                #     'let': d[0].to_dict()["Lat"],
                                #     'long': d[0].to_dict()["long"],
                                #     'time': d[0].to_dict()["time"],
                                #     'pic': "images"+ str(c) + ".jpg",
                                #     'class': labels[class_current],
                                #     'path': t
                                #         }) 

                        bounding_boxes.append([x_min, y_min, x_max, y_max])

                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)

   
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                    probability_minimum, threshold)

            if len(results) > 0:
              
                for i in results.flatten():
             
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    x_max, y_max = bounding_boxes[i][2], bounding_boxes[i][3]

               
                    colour_box_current = colours[class_numbers[i]].tolist()

                    cv2.rectangle(frame, (x_min, y_min),
                                (x_max, y_max),
                                colour_box_current, 2)
       
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                        confidences[i])

                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

            cv2.namedWindow('YOLO v3 Real Time Detections')
            cv2.imshow('YOLO v3 Real Time Detections', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        camera.release()
        cv2.destroyAllWindows()




################################## Main ###############################################




detect()
car_ref = db.collection('car')
location = db.collection('locations')
docs = car_ref.stream()
docsc = car_ref.stream()
docsl = location.stream()
# y = json.loads(docs)
for doc in docs:
    
    time = doc.to_dict()["time"]
    # print(time)

    doc2 = location.where('time','==',time).stream()
    d = list(doc2)
    if d != []:
    
        # print(d[0].to_dict()["time"])
        
        doc_ref2 = db.collection('map4').document()
        doc_ref2.set({
        'let': d[0].to_dict()["Lat"],
        'long': d[0].to_dict()["long"],
        'time': d[0].to_dict()["time"],
        'pic': doc.to_dict()["pic"],
        'class': doc.to_dict()["class"],
        'path': doc.to_dict()["path"]
        }) 
for doc1 in docsc:
    print(f'Deleting doc {doc1.id} => {doc1.to_dict()}')
    doc1.reference.delete()

for doc3 in docsl:
    # print(f'Deleting doc {doc3.id} => {doc3.to_dict()}')
    doc3.reference.delete()

feature_group = FeatureGroup(name="mend")
feature_group1 = FeatureGroup(name="crack")
feature_group2 = FeatureGroup(name="waterpipecap")
me = 0
cr = 0
wa = 0
location = db.collection('map4')
docs = location.stream()
with open('map.csv','w', newline='') as f:
    fieldnames =['LAT','LONG','TIME','PIC','TYPE']
    thewriter = csv.DictWriter(f,fieldnames=fieldnames)
    for doc in docs:
        
        LAT = doc.to_dict()["let"]
        LOG = doc.to_dict()["long"]
        localtime = doc.to_dict()["time"]
        picture = doc.to_dict()["path"]
        type1 = doc.to_dict()["class"]
        name = doc.to_dict()["pic"]
        thewriter.writerow({'LAT': LAT,'LONG': LOG,'TIME': localtime,'PIC':picture,'TYPE':type1 })
        if type1=='waterpipecap':
                folium.Marker(location=[LAT,LOG],popup="<b>name: </b> "+name+"<br> <b>LAT: </b> "+str(LAT)+"<br> <b>LOG: </b> "+str(LOG)+"<br><b>Class: </b> "+type1+"<br> <b>Time : </b> "+localtime+"<br> <a href="+picture+"><img src="+picture+" height=142 width=290></a>",icon=folium.Icon(color='green')).add_to(feature_group2)
                wa+=1
        elif type1=='mend':
                folium.Marker(location=[LAT,LOG],popup="<b>name: </b> "+name+"<br> <b>LAT: </b> "+str(LAT)+"<br> <b>LOG: </b> "+str(LOG)+"<br><b>Class : </b> "+type1+"<br> <b>Time : </b> "+localtime+"<br> <a href="+picture+"><img src="+picture+" height=142 width=290></a>",icon=folium.Icon(color='blue')).add_to(feature_group)
                me+=1 
        elif type1=='cracks':
                folium.Marker(location=[LAT,LOG],popup="<b>name: </b> "+name+"<br> <b>LAT: </b> "+str(LAT)+"<br> <b>LOG: </b> "+str(LOG)+"<br><b>Class : </b> "+type1+"<br> <b>Time : </b> "+localtime+"<br> <a href="+picture+"><img src="+picture+" height=142 width=290></a>",icon=folium.Icon(color='red')).add_to(feature_group1)
                cr+=1
        
 

m =Map(location=[LAT,LOG],zoom_start=80)
feature_groupcount = FeatureGroup(name="mend"+str(me))
feature_groupcount1 = FeatureGroup(name="crack"+str(cr))
feature_groupcount2 = FeatureGroup(name="waterpipecap"+str(wa))
feature_group.add_to(m)
feature_group1.add_to(m)
feature_group2.add_to(m)
feature_groupcount.add_to(m)
feature_groupcount1.add_to(m)
feature_groupcount2.add_to(m)
LayerControl().add_to(m)

m.save("test9.html")

