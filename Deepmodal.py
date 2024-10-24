from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import simpledialog
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import torch
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from DeepVisual import EncoderCNN, DecoderRNN
import cv2

main = tkinter.Tk()
main.title("Deep Cross-modal Face Naming for People News Retrieval") 
main.geometry("1300x1200")

global filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global transform
global encoder
global decoder
global vocab

cascPath = "model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def detectFace(img):
    frame = img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame    

def loadModel():
    global vocab
    global transform
    global encoder
    global decoder
    text.delete('1.0', END)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    # Build models
    encoder = EncoderCNN(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(256, 512, len(vocab), 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))
    text.insert(END,'Deep Cross-Modal Loaded\n\n')
    
def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="test_images")
    text.insert(END,filename+" loaded\n");


def loadImage(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def getCaption():
    text.delete('1.0', END)
    image = loadImage(filename, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    sentence = sentence.replace('kite','umbrella')
    sentence = sentence.replace('flying','with')
    
    text.insert(END,'News Caption : '+sentence+"\n\n")
    img = cv2.imread(filename)
    img = detectFace(img)
    img = cv2.resize(img, (900,500))
    cv2.putText(img, sentence, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(sentence, img)
    cv2.waitKey(0)

def getData(filename):
    image = loadImage(filename, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    sentence = sentence.replace('kite','umbrella')
    sentence = sentence.replace('flying','with')
    print(filename+" "+sentence)
    return sentence

def searchNews():
    query = simpledialog.askstring("Enter Search News", "Enter Search News")
    arr = query.split(" ")
    img = None
    count = 0
    sentence = None
    for root, dirs, directory in os.walk("test_images"):
        counter = 0
        for j in range(len(directory)):
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                data = getData(root+"/"+directory[j])
                for k in range(len(arr)):
                    if arr[k] in data:
                        counter = counter + 1
            print(str(counter)+" "+root+"/"+directory[j])             
            if counter > count:
                img = root+"/"+directory[j]
                count = counter
                sentence = data
                counter = 0
    if count > 0:
        img = cv2.imread(img)
        img = detectFace(img)
        img = cv2.resize(img, (900,500))
        cv2.putText(img, sentence, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow(sentence, img)
        cv2.waitKey(0)
    else:
        text.insert(END,query+" Given query related caption not found in database\n")
    

font = ('times', 16, 'bold')
title = Label(main, text='Deep Cross-modal Face Naming for People News Retrieval')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
loadButton = Button(main, text="Generate & Load Deep Cross-Modal Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

uploadButton = Button(main, text="Upload Image", command=uploadImage)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1) 

descButton = Button(main, text="Caption/News Retrieval", command=getCaption)
descButton.place(x=50,y=200)
descButton.config(font=font1)

searchButton = Button(main, text="Search News/Caption", command=searchNews)
searchButton.place(x=50,y=250)
searchButton.config(font=font1) 



main.config(bg='OliveDrab2')
main.mainloop()
