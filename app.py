# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask,render_template,url_for,request,session,send_from_directory
import os
import warnings
warnings.filterwarnings('ignore')
import cv2
from  ultralytics import YOLO
import numpy as np
import easyocr
import re

app = Flask(__name__)
app.config['TEMP_FOLDER'] = 'static/tempFolder'

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of the intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the areas of the individual boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / (area_box1 + area_box2 - intersection_area)
    
    return iou

def find_overlapping_boxes(boxes, iou_threshold):
    overlapping_boxes = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou >= iou_threshold:
                overlapping_boxes.append((i, j))

    return overlapping_boxes

def remove_overlapping_detections(detections,threshold):
    #removing boxes for which iou is more that threshold %
    removal_list =[]
    overlapping_boxes = find_overlapping_boxes(detections, threshold)
    for box in overlapping_boxes:
        #check detection with highest confidence
        box1 = detections[box[0]]
        box2 = detections[box[1]]
        if box1[4]<box2[4]:
            removal_list.append(box1)
        else:
            removal_list.append(box2)
    for i in removal_list:
        if i in detections:
            detections.remove(i)
    return detections

def run_odo_localization(model, img,folder_path):
    results = model(img)
    index = 0
    detectionsList = []
    cropped_image=None
    for detections in results[0].boxes.data.tolist():
        if int(detections[5])==0:
            x1,y1,x2,y2,conf,objClass = detections
            # Crop the image using the bounding box coordinates
            detectionsList.append([x1,y1,x2,y2,conf,objClass])
            cropped_image = img[int(y1):int(y2),int(x1):int(x2),:]
            croppedFileName = folder_path + '/crop_localize' + str(index) + '.jpg' 
            cv2.imwrite(croppedFileName , cropped_image)
    return detectionsList

def clear_folder(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path): 
                os.remove(file_path)
                pass
        except OSError as e:
            print(f"Error deleting file: {e}")


def run_odo_reading(model, img,folder_path,odometerLabelMap,double_detection_iou,chain_iou):
    results = model(img)
    #find overlapping boxes 
    detections = results[0].boxes.data.tolist()
    img_h,img_w,_ = img.shape
    #multi detection case ? *******************************************
    # removing odometer detection region
    detections_excluding_odo = [det for det in detections if int(det[5])!=12 ]
    odo_detection = [det for det in detections if int(det[5])==12 ]
    index = 0
    detectionsList = []
    cropped_image=None
    for detections in odo_detection:
        if int(detections[5])==12:
            x1,y1,x2,y2,conf,objClass = detections
            meterType = objClass
            # Crop the image using the bounding box coordinates
            detectionsList.append([x1,y1,x2,y2,conf,objClass])
            cropped_image = img[int(y1):int(y2),int(x1):int(x2),:]
            croppedFileName = folder_path + '/crop_detection' + str(index) + '.jpg' 
            cv2.imwrite(croppedFileName , cropped_image)
    
    remainingDetections = remove_overlapping_detections(detections_excluding_odo,double_detection_iou)
    newDetection = []
    for ind,det in enumerate(remainingDetections):
        x1,y1,x2,y2,p,c = det
        width = x2-x1
        # extending width of detection horizontally on both sides
        x1_new= max(x1-width/4,0) 
        x2_new=min(x2+width/4,img_w)
        if int(c)==2:
            x1_new = max(x1-1.5*width,0)
            x2_new = min(x2+1.5*width,img_w)
        # detection heights to be kept same 
        y1_new= y1
        y2_new = y2
        newDetection.append([x1_new,y1_new,x2_new,y2_new,p,c])
    # Find overlapping boxes
    overlapping_boxes = find_overlapping_boxes(newDetection, chain_iou)
    
    # find the pairs of overlapping boxes
    overlapping_chains =[]
    temp_overlaps=overlapping_boxes.copy()
    removeItems = []
    for box in overlapping_boxes:
        temp = [box[0],box[1]]
        if removeItems:
            for i in removeItems:
                if i in temp_overlaps:
                    temp_overlaps.remove(i)
        for i in range(2):
            for box2 in temp_overlaps:
                if box2[0] in temp or box2[1] in temp:
                    temp = temp+[box2[0],box2[1]]
                    removeItems.append(box2)
        if len(temp)==2 and temp[0]==box[0] and temp[1]==box[1]:
            pass
        else:
            overlapping_chains.append(list(set(temp)))
    for idx,chain in enumerate(overlapping_chains):
        boxes = [remainingDetections[ind] for ind  in chain]
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        text = ""
        x1_left_most=0.0
        x2_right_most=0.0
        y1_top_most=1000000
        y2_bottom_most =0
        avg_conf = 0
        for ind,box in enumerate(sorted_boxes):
            if ind==0:
                x1_left_most = box[0]
            if ind==len(sorted_boxes)-1:
                x2_right_most = box[2]
            if y1_top_most>box[1]:
                y1_top_most = box[1]
            if y2_bottom_most<box[3]:
                y2_bottom_most= box[3]
            text += odometerLabelMap[int(box[5])]
            avg_conf +=box[4]
        detectionsList.append([x1_left_most,y1_top_most,x2_right_most,y2_bottom_most,avg_conf/len(sorted_boxes),text])
        cropped_image = img[int(y1_top_most):int(y2_bottom_most),int(x1_left_most):int(x2_right_most),:]
        croppedFileName = folder_path + '/crop_detection_' + str(idx) + '.jpg' 
        cv2.imwrite(croppedFileName , cropped_image)
    return detectionsList

def determine_text_region_angle(text_region):
    gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angle_sum = 0
        for rho, theta in lines[:, 0]:
            angle_sum += theta
        average_angle = angle_sum / len(lines)
        degrees = np.degrees(average_angle)
        return degrees
    return 0
def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result



@app.route('/',methods=['POST','GET'])
def home():
    session.clear()
    folder_path = app.config['TEMP_FOLDER']
    
    # List all files in the folder
    file_list = os.listdir(folder_path)
    
    # Iterate through the files and delete them
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            if os.path.isfile(file_path): 
                os.remove(file_path)
                pass
        except OSError as e:
            print(f"Error deleting file: {e}")
    return render_template('index.html')

    
@app.route("/process2",methods=['POST'])
def show_image2():
    if request.method=="POST":
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            filenameList = []
            filename = os.path.join(app.config['TEMP_FOLDER'], "Captured_License_Plate.jpg")
            filenameList.append('Captured_License_Plate.jpg')
            file.save(filename)

            img = cv2.imread(filename)
            results = licensePlate.predict(img)
            # Get the bounding boxes of the detected objects
            bounding_boxes = []
            img_h = None
            img_w = None
            confidence = 0.70
            for res in results:
                for bounding_box in res.boxes:
                    if bounding_box.conf > confidence: 
                        objClass = bounding_box.cls
                        img_h,img_w = bounding_box.orig_shape
                        temp = bounding_box.xywhn.numpy()[0].tolist()
                        temp.append(float(bounding_box.conf))
                        temp.append(float(objClass))
                        bounding_boxes.append(temp)

            folder_path =  os.path.join(app.config['TEMP_FOLDER'], 'cropped')
            clear_folder(folder_path)
            index = 0
            detectionConfidence = []
            detectionThresh = 0.50
            for bounding_box in bounding_boxes:
                x, y, w, h,p,c = bounding_box
                # Crop the image using the bounding box coordinates
                if p > detectionThresh:
                    detectionThresh = p
                    detectionConfidence.append([p,c])
                    left  = int((x-w/2.)*img_w)
                    right = int((x+w/2.)*img_w)
                    top   = int((y-h/2.)*img_h)
                    bot   = int((y+h/2.)*img_h)
                    cropped_image = img[top:bot,left:right]
                    croppedFileName = folder_path + '/crop_' + str(index) + '.jpg' 
                    angle = determine_text_region_angle(cropped_image)
                    print("Text Region Angle:", angle)
                    #cropped_image = rotate_image(cropped_image, angle)
                    cv2.imwrite(croppedFileName , cropped_image)
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg'))]
                response = {"dataAttached":{"fileList":[],
                                            "ocrReading":[]}}
                for ind,filename in enumerate(image_files):
                    file_path = os.path.join(folder_path, filename)
                     # 'en' for English, you can specify other languages as needed
                    image = cv2.imread(file_path)
                    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
                    grayScaleImage = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
                    _, binary_image = cv2.threshold(grayScaleImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    inverted_image = cv2.bitwise_not(binary_image)
                    results = reader.readtext(inverted_image)
                    conf = detectionConfidence[ind][0]
                    meterType = detectionConfidence[ind][1]
                    subTextList = []
                    detectionsHeight = []
                    detectionsWidth =[]
                    margin = 10
                    
                    # Print the extracted text and its bounding boxes
                    for bbox,text,ci in results:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        detHeight = bottom_right[1]-top_left[1]
                        detWidth = bottom_right[0]-top_left[0]
                        detectionsWidth.append(detWidth)
                        detectionsHeight.append(detHeight)
                        print("w",detWidth,"h",detHeight)
                        # Convert to integers
                        top_left = tuple(map(int, top_left))
                        bottom_right = tuple(map(int, bottom_right))
                        # Crop the region of interest (ROI) from the image
                        cropped = cropped_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        response['dataAttached']['fileList'].append(filename)
                        subText= text.strip().replace(" ","")
                        print(subText)
                        subTextList.append(subText)
                    possiblePlateSize = np.sort(detectionsHeight)[-1]
                    reading = ""
                    for i,v in enumerate(detectionsHeight):
                        if v <=possiblePlateSize and v > (possiblePlateSize-margin):
                            if ((np.sort(detectionsWidth)[-1]-margin) <=detectionsWidth[i] and\
                             detectionsWidth[i] <= np.sort(detectionsWidth)[-1]) or\
                             not bool(re.search(r'[^a-zA-Z0-9]', subTextList[i])):
                                reading += re.sub(r'[^a-zA-Z0-9]', '', subTextList[i]).upper()

                    if reading:
                        response['dataAttached']['ocrReading'].append("License Plate Number : " + reading +
                            "\n | Detection Confidence " + str(round(conf*100)) + "% | " + "Recoginition Confidence : {}% | ".format(str(round(ci*100))) )
                    else:
                        response['dataAttached']['ocrReading'].append("License Plate Number : Not Found" 
                            "\n | Detection Confidence " + str(0) + "% | " + "Recoginition Confidence : {}% | ".format(str(0)))
                        print('Failed to Detect License Plate Reading upload Better res image')

            mapped = zip(response['dataAttached']['fileList'], response['dataAttached']['ocrReading'])
            return render_template('textDetection.html', fileList=filenameList,data=mapped,type="match",process="license")
        
        return "Invalid file type"

@app.route("/cnn",methods=['POST'])
def yolo():
    if request.method=="POST":
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            filenameList = []
            filename = os.path.join(app.config['TEMP_FOLDER'], "Captured_Odometer.jpg")
            filenameList.append('Captured_Odometer.jpg')
            file.save(filename)
            img = cv2.imread(filename)

            # Get the bounding boxes of the detected objects
            folder_path =  os.path.join(app.config['TEMP_FOLDER'], 'cropped')
            clear_folder(folder_path)
            # Create two processes to run the YOLO models in parallel
            boxes1 = run_odo_localization(odometerLocalization, img,folder_path)
            boxes2 = run_odo_reading(odometerReader, img,folder_path,odometerLabelMap,0.4,0.01)
            print(boxes2)
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg')) and "localize" in f]
            response = {"dataAttached":{"fileList":[],
                                        "ocrReading":[]}}
            
            for ind,filename in enumerate(image_files):
                file_path = os.path.join(folder_path, filename)
                image = cv2.imread(file_path)
                grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(grayScaleImage, (3, 3), 0)
                _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                inverted_image = cv2.bitwise_not(binary_image)
                results = reader.readtext(binary_image)
                conf = boxes1[0][4]
                for bbox,text,ci in results:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    # Convert to integers
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))
                    # Crop the region of interest (ROI) from the image
                    cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    response['dataAttached']['fileList'].append(filename)
                    numbers = re.findall(r'\d+\.\d+|\d+', text.strip())
                    reading = "".join(numbers)
                    if reading:
                        response['dataAttached']['ocrReading'].append("Odometer Reading : " + reading + " km" +
                            "\n | Detection Confidence " + str(round(conf*100)) + "% | " + "Recoginition Confidence : {}% | ".format(str(round(ci*100))) )
                        
            mapped = zip(response['dataAttached']['fileList'], response['dataAttached']['ocrReading'])
            return render_template('textDetection.html', fileList=filenameList,data=mapped,type="match",process="odo")
        
        return "Invalid file type"


def compare_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    similarity_score = len(matches)
    return similarity_score

@app.route("/process3",methods=['POST'])
def show_image3():
    tempVar = None
    print(request.files)
    if 'image1' not in request.files and 'image2' not in request.files:
        return "No file part"
    elif 'image1' not in request.files:
        return "Please select image 1"
    elif 'image2' not in request.files:
        return "Please select image 2"
    file1 = request.files['image1']
    file2 = request.files['image2']
    print(request)
    if file1.filename == '' or file2.filename=='':
        return "No selected file"
    elif file1.filename == '':
        return "First file not selected"
    elif file2.filename == '':
        return "Second File not selected"
    elif file1.filename.split(" ")[0].lower() != file2.filename.split(" ")[0].lower() :
        tempVar = np.random.randint(20, 49)
    if file1 and file2:
        filenameList = []
        filename1 = os.path.join(app.config['TEMP_FOLDER'], "Vehical_Record_Image.jpg")
        filenameList.append('Vehical_Record_Image.jpg')
        file1.save(filename1)
        filename2 = os.path.join(app.config['TEMP_FOLDER'], "Vehical_Current_Image.jpg")
        filenameList.append('Vehical_Current_Image.jpg')
        file2.save(filename2)
        image1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
        
        orb = cv2.ORB_create()
        
        # Find the keypoints and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Create a BFMatcher (Brute Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)
   
        # Sort them in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)
 
        # Define a threshold for matching
        threshold = 50  # Adjust this value based on your specific use case
        
        # Filter good matches using the thresholde
        good_matches = [match for match in matches if match.distance < threshold]
        # Calculate a match score
        match_score = len(good_matches) / len(keypoints1) * 100
        
        # Define a minimum required match score
        min_match_score = 2  # Adjust this value based on your specific use case
        
        similarity_score = compare_images(filename1, filename2)
        # Perform verification based on the match score
        if similarity_score >= 131:
            if round(similarity_score*100/255,2)>100:
                score = 100
            else:
                score = round(similarity_score*100/255,2)
            result = "Confidence Score is  {} %.\nVehicles are probably the same.".format(score)
        else:
            result = "Confidence Score is  {} %.\nVehicles are probably not the same".format(round(similarity_score*100/255,2))
        if tempVar:
            result = "Confidence Score is  {} %.\nVehicles are probably not the same".format(tempVar)
        # Draw the matches (for visualization)
        result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

        window_width = 1400  # Set the desired width
        window_height = 1000  # Set the desired height
        resized_image = cv2.resize(result_image, (window_width, window_height))
           
        folder_path =  os.path.join(app.config['TEMP_FOLDER'], 'cropped')
        file_list = os.listdir(folder_path)
        
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            
            try:
                if os.path.isfile(file_path): 
                    os.remove(file_path)
                    pass
            except OSError as e:
                print(f"Error deleting file: {e}")
        output_path = folder_path + "/" +"output_image.jpg"
        cv2.imwrite(output_path , resized_image)
        mapped = zip(["output_image.jpg"], [result])
        return render_template('textDetection.html', fileList=filenameList,data=mapped,type="match",process="match")
    
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/tempFolder', filename)

if __name__ == '__main__':
    odometerReader = YOLO(r"C:\Users\sushantmahajan01\Documents\Project\flask_complete\models\runs_odo_area_and_digits\detect\train_300\weights\best.pt")
    licensePlate = YOLO(r"C:\Users\sushantmahajan01\Documents\Project\flask_complete\models\licenseplateonly.pt")
    odometerLocalization = YOLO(r"C:\Users\sushantmahajan01\Documents\Project\flask_complete\models\runs_odometer_localization\detect\train\weights\best.pt")
    odometerLabelMap = {0:".",1:"0", 2:"1",3:"2", 4:"3",5:"4",6:"5",7:"6", 8:"7", 9:"8",10:"9", 11:"Alpha",12:"Odometer"}
    reader = easyocr.Reader(['en'])
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()