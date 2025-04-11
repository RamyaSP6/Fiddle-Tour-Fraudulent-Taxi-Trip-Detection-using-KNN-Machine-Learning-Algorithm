import numpy
import tflearn
import os
import subprocess
import numpy
import random
import json
import pickle
from time import sleep
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox 
# import filedialog module
from tkinter import filedialog
flg=0;
import tkinter as tk
model1 = pickle.load(open('training_pickle', 'rb'))
 

def checkTaxifraud():
    in1 = inputtxt1.get();
    in2 = inputtxt2.get();
    in3 = inputtxt3.get();
    in4 = inputtxt4.get();
    in5 = inputtxt5.get();
    in6 = inputtxt6.get();
    main = in1 + " " + in2 + " " + in3 + " " + in4 + " " + in5 + " " + in6
    userList = main.split()
    print("user list is ", userList)

    list_of_floats = [float(item) for item in userList]

    print(list_of_floats)
    import numpy as np
    ynew=model1.predict(np.array(list_of_floats).reshape(1, -1))
    ynew = ynew.round()
    #ynew = ynew.argmax()
    print(ynew)
    if (ynew[0]==0):
        print(" ")
        print("Normal")
        messagebox.showinfo('Result', 'Normal')
        print(" ")
    if (ynew[0]==1):
        print("  ")
        print("Taxi Fraud Detected")
        messagebox.showinfo('Result', 'Taxi Fraud Detected')
        print(" ")

t=0;
if(t==0):
    #if __name__ == '__main__':
    print("started")
    window = Tk()
  
    # Set window title
    window.title('Taxi Fault Detection system')
      
    # Set window size
    window.geometry("700x600")
      
    #Set window background color
    window.config(background = "white")
      
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "Please fill Inputs",
                                width = 100, height = 4,
                                fg = "blue")
      
    lbl1 = tk.Label(window, text = "Pick up Longitude")
    inputtxt1 = Entry(window)

    lbl2 = tk.Label(window, text = "Pick up Latitude")
    inputtxt2 = Entry(window)

    lbl3 = tk.Label(window, text = "Drop off Longitude")
    inputtxt3 = Entry(window)

    lbl4 = tk.Label(window, text = "Drop off Latitude")
    inputtxt4 = Entry(window)
    
    lbl5 = tk.Label(window, text = "Passenger Count")
    inputtxt5 = Entry(window)
    
    lbl6 = tk.Label(window, text = "Amount")
    inputtxt6 = Entry(window)

    
    button_start = Button(window,
                         text = "submit", command = checkTaxifraud)

       
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1, padx=5, pady=5)
    
    lbl1.grid(column = 1, row = 2, padx=5, pady=5)
    inputtxt1.grid(column = 1, row = 3, padx=5, pady=5)
    lbl2.grid(column = 1, row = 8, padx=5, pady=5)
    inputtxt2.grid(column = 1,row = 9, padx=5, pady=5)
    lbl3.grid(column = 1, row = 11, padx=5, pady=5)
    inputtxt3.grid(column = 1,row = 12, padx=5, pady=5)
    lbl4.grid(column = 1, row = 15, padx=5, pady=5)
    inputtxt4.grid(column = 1,row = 16, padx=5, pady=5)
    lbl5.grid(column = 1, row = 18, padx=5, pady=5)
    inputtxt5.grid(column = 1,row = 20, padx=5, pady=5)
    lbl6.grid(column = 1, row = 22, padx=5, pady=5)
    inputtxt6.grid(column = 1,row = 24, padx=5, pady=5)
    button_start.grid(column = 1,row = 26, padx=5, pady=5)
      
    # Let the window wait for any events
    
    
    window.mainloop()

    #checkTaxifraud()
