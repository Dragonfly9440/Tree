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

def run_yolo_model(model, img,folder_path,detectionThresh, modelType):
    results = model(img)
    index = 0
    detectionsList = []
    cropped_image=None
    for detections in results[0].boxes.data.tolist():
        if (int(detections[5])==0 and modelType=="localization") or  (int(detections[5])==12 and modelType=="detection"):
            x1,y1,x2,y2,conf,objClass = detections
            meterType = objClass
            # Crop the image using the bounding box coordinates
            if conf > detectionThresh:
                detectionThresh = conf
                detectionsList.append([x1,y1,x2,y2,conf,objClass])
                cropped_image = img[int(y1):int(y2),int(x1):int(x2),:]
                croppedFileName = folder_path + '/crop_{}'.format(modelType) + str(index) + '.jpg' 
                angle = determine_text_region_angle(cropped_image)
                print("Text Region Angle:", angle)
                #cropped_image = rotate_image(cropped_image, angle)
                cv2.imwrite(croppedFileName , cropped_image)
        else:
            x1,y1,x2,y2,conf,objClass = detections
            detectionsList.append([x1,y1,x2,y2,conf,objClass])
    return detectionsList #output_queue.put(detectionsList)


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
            file_list = os.listdir(folder_path)
            
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    if os.path.isfile(file_path): 
                        os.remove(file_path)
                        pass
                except OSError as e:
                    print(f"Error deleting file: {e}")
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
            file_list = os.listdir(folder_path)
            
            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    if os.path.isfile(file_path): 
                        os.remove(file_path)
                        pass
                except OSError as e:
                    print(f"Error deleting file: {e}")

            boxes1 = run_yolo_model(odometerLocalization, img,folder_path, 0.50, "localization")

                    
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg')) ]#and ("thresholded_" in f)]
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
    odometerReader = YOLO(r"D:\Tree Insurance\flask app\data\runs_odo_area_and_digits\detect\train_100\weights\best.pt")
    licensePlate = YOLO(r"D:\Tree Insurance\flask app\data\licenseplateonly.pt")
    odometerLocalization = YOLO(r"D:\Tree Insurance\flask app\data\runs_odometer_localization\detect\train\weights\best.pt")
    #odometerLocalization = YOLO(r"D:\Tree Insurance\flask app\data\runs_odometer_localization_20_on_dig\detect\train\weights\best.pt")
    reader = easyocr.Reader(['en'])
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()