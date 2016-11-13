import cv2
import  os
from feature import Fisherfaces
from model import PredictableModel,NearestNeighbor,EuclideanDistance
import numpy as np
import cPickle
import pyodbc as p
con= p.connect('DRIVER={SQL Server Native Client 11.0}',Server='(localdb)\ProjectsV12',DATABASE='Python link')

# print"Please enter the name of person and press 'ENTER'keyword to find the location "
# testVar = raw_input("Name:")

def save_model(filename, model):
    output = open(filename, 'wb')
    cPickle.dump(model, output)
    output.close()

def load_model(filename):
    pkl_file = open(filename, 'rb')
    res = cPickle.load(pkl_file)
    pkl_file.close()
    return res

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image):
        min_size = (30, 30)
        faces_coord = self.classifier.detectMultiScale(image,1.3,5,minSize=min_size)
        return faces_coord

class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print self.video.isOpened()

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        ret, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
    return faces

def normalize_intensity(images):
    images_norm=[]
    for image in images:
        is_color=len(image.shape)==3
        if is_color:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def apply_filter(images):
    images_norm=[]
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.GaussianBlur(image,(5,5),0))
    return images_norm

def resize(images, size=(64,64)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces =apply_filter(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h),
                              (150, 150, 0), 3)

def collect_dataset():
    X = []#image vector
    y = []#label  eg 1,2,3
    labels_dic = {}
    people = [person for person in os.listdir("Member/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("Member/" + person):
            X.append(cv2.imread("Member/" + person + '/' + image, 0))
            y.append(i)
    return (X, np.array(y), labels_dic)

X, y, labels_dic = collect_dataset()

feature = Fisherfaces()

classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=3)

model = PredictableModel(feature=feature, classifier=classifier)
    # Compute the Fisherfaces on the given data (in X) and labels (in y):
model.compute(X, y)
save_model('model.pkl', model)
print "Models Trained Succesfully"
model = load_model('model.pkl')
detector = FaceDetector("haarcascade_frontalface_default.xml")
webcam = VideoCamera(0)
cv2.namedWindow("Face_Recognition",cv2.WINDOW_AUTOSIZE)

def main_process(face):

    prediction=model.predict(face)
    pred = prediction[0]
    conf = prediction[1]
    distance = conf['distances'][0]
    if distance<700:

        a = labels_dic[pred]
        print a
        cur = con.cursor()
        querystring = "select Searchname from Link"
        cur.execute(querystring)
        for row in cur:
            b=row[0]
        cur.commit()

        if a==b:
                cur4 = con.cursor()
                querystring4 = "Update Link Set location='Camera 1' where Id='True' "
                cur4.execute(querystring4)
                querystring5 = "select location from Link"
                cur4.execute(querystring5)
                for row in cur4:
                    z=row[0]
                print z
                cur4.commit()

                if(z=="Camera 1"):
                    cur2 = con.cursor()
                    querystring3 = "Update Link Set Updated='True' where Id='True' "
                    cur2.execute(querystring3)
                    cur2.commit()
                    cur3 = con.cursor()
                    querystring2 = "Select Updated from Link"
                    cur3.execute(querystring2)
                    for row in cur3:
                        c=row[0]
                    cur3.commit()
                else:
                    cur2 = con.cursor()
                    querystring3 = "Update Link Set Updated='True' where Id='True' "
                    cur2.execute(querystring3)
                    cur2.commit()


while True:
    cur1 = con.cursor()
    querystring1 = "select Start from Link"
    cur1.execute(querystring1)
    for row in cur1:
        b=row[0]
    while b == False:
        cur1.execute(querystring1)
        for row in cur1:
            b=row[0]
    cur1.commit()
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            main_process(face)
    cv2.imshow("Face_Recognition", frame)
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break;