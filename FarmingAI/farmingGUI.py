import tkinter as tk
from tkinter import * 
from getModel import computeModel

# check if string is number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

root = Tk()

# This is the section of code which creates the main window
root.geometry('1000x520')
root.configure(background='#90EE90')
root.title('FarmingAI')


filename = PhotoImage(file = 'sss.png')
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# This is the section of code which creates the a label
Label(root, text='Welcome to FarmingAI!', bg='lemon chiffon', font=('arial', 18, 'bold')).place(x=400, y=13)
Label(root, text='Please input your values below in the form of numbers only,\nthe AI will use it to give you the best crop to plant:', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=56)
Label(root, text='Nitrogen content ratio in soil:', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=123)
Label(root, text='Phosphorus content ratio in soil:', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=163)
Label(root, text='Potassium content ratio in soil:', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=203)
Label(root, text='Temperature (in C):', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=243)
Label(root, text='Humidity (%):', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=283)
Label(root, text='ph Level:', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=323)
Label(root, text='Rainfall (mm):', bg='lemon chiffon', font=('arial', 15, 'normal')).place(x=18, y=363)
# This is the section of code which creates a text input box
nitrogenInput1 = StringVar()
phosInput1 = StringVar()
potaInput1 = StringVar()
rainInput1 = StringVar()
nitrogenInput1.set("")
phosInput1.set("")
potaInput1.set("")
rainInput1.set("")

nitrogenInput=Entry(root, font = ('arial', 15, 'normal'), width = 11, textvariable = nitrogenInput1)
nitrogenInput.place(x=358, y=123)


# This is the section of code which creates a text input box
phosInput=Entry(root, width = 11, font = ('arial', 15, 'normal'), textvariable = phosInput1)
phosInput.place(x=358, y=163)

# This is the section of code which creates a text input box
potaInput=Entry(root, width = 11, font = ('arial', 15, 'normal'), textvariable = potaInput1)
potaInput.place(x=358, y=203)


def correct_input_NPKR():
    if float(nitrogenInput1.get()) < 0 :
         nitrogenInput1.set('0')  
    if float(phosInput1.get()) < 0 :
         phosInput1.set('0')
    if float(potaInput1.get()) < 0 :
         potaInput1.set('0')
    if float(rainInput1.get()) < 0 :
         rainInput1.set('0')
        

def correct_input_T(text):
    valid = False
    try:
        float(text)
        if float(text) >= -40 and float(text)<= 60:
            return True
        else:
            if float(text)>= 0: 
                Tvar.set('60')
            else:
                Tvar.set('-40')
                
            return False
            
    except:
        errorLabelstr.set('Please input numbers only.')
        return False

def correct_input_H(text):
    valid = False
    try:
        float(text)
        if float(text) >= 0 and float(text)<= 100:
            return True
        else:
            if float(text)>= 0: 
                Hvar.set('100')
            else:
                Hvar.set('0')
                
            return False
            
    except:
        errorLabelstr.set('Please input numbers only.')
        return False

def correct_input_pH(text):
    valid = False
    try:
        float(text)
        if float(text) >= 1 and float(text)<= 14:
            return True
        else:
            if float(text)>= 0: 
                pHvar.set('14')
            else:
                pHvar.set('1')
                
            return False
            
    except:
        errorLabelstr.set('Please input numbers only.')
        return False

# This is the section of code which creates a text input box
validate_input = (root.register(correct_input_T), '%P')
Tvar = StringVar(root)
Tvar.set("0")
tempInput= Spinbox(root, from_=-40, to=60,  validate = 'all', validatecommand = validate_input, textvariable=Tvar, font=('arial', 15, 'normal'), width=10)
tempInput.place(x=358, y=243)

# This is the section of code which creates a text input box
validate_input = (root.register(correct_input_H), '%P')
Hvar = StringVar(root)
Hvar.set("0")
humInput= Spinbox(root, from_=0, to=100,textvariable=Hvar,validate = 'all', validatecommand = validate_input,  font=('arial', 15, 'normal') , width=10)
humInput.place(x=358, y=283)

# This is the section of code which creates a text input box
validate_input = (root.register(correct_input_pH), '%P')

pHvar = StringVar(root)
pHvar.set("7")
phSpinBox= Spinbox(root, from_=1, to=14,textvariable=pHvar, validate = 'all', validatecommand = validate_input, font=('arial', 15, 'normal'), bg = None, width=10)
phSpinBox.place(x=358, y=323)

# This is the section of code which creates a text input box
rainInput=Entry(root, font = ('arial', 15, 'normal'), width =11, textvariable = rainInput1)
rainInput.place(x=358, y=363)


# this is the function called when the button is clicked
def farmBttnClicked():
    canwork = True
    isNum = True
    if len(rainInput.get())==0 or len(phSpinBox.get())==0 or len(humInput.get())==0 or len(tempInput.get())==0:
        canwork=False
    if len(potaInput.get())==0 or len(phosInput.get())==0 or len(nitrogenInput.get())==0 :
        canwork=False
    if is_number(rainInput.get())==False or is_number(phSpinBox.get())==False or is_number(humInput.get())==False or is_number(tempInput.get())==False:
        isNum = False
    if is_number(potaInput.get())==False or is_number(phosInput.get())==False or is_number(nitrogenInput.get())==False :
        isNum =False
    
    if canwork==False:
        errorLabelstr.set('Please fill in all the fields.')
    if canwork==True and isNum == False:
        errorLabelstr.set('Please only enter numbers in the fields.')
    if isNum==True and canwork==True:
        errorLabelstr.set('') 
        correct_input_T(tempInput.get())
        correct_input_H(humInput.get())
        correct_input_pH(phSpinBox.get())
        correct_input_NPKR()
        x = []
        x.append(nitrogenInput.get())
        x.append(phosInput.get())
        x.append(potaInput.get())
        x.append(tempInput.get())
        x.append(humInput.get())
        x.append(phSpinBox.get())
        x.append(rainInput.get())
        answerStr.set(computeModel(x))
        #answerStr.set("computing works")
        if answerStr.get() == 'muskmelon':
            canvas.itemconfig(image_id, image=muskMelonPic)
        elif answerStr.get() == 'mango':
            canvas.itemconfig(image_id, image=mangoPic)
        elif answerStr.get() == 'apple':
            canvas.itemconfig(image_id, image=applePic)
        elif answerStr.get() == 'grapes':
            canvas.itemconfig(image_id, image=grapesPic)
        elif answerStr.get() == 'orange':
            canvas.itemconfig(image_id, image=orangePic)
        elif answerStr.get() == 'banana':
            canvas.itemconfig(image_id, image=bananaPic)
        elif answerStr.get() == 'chickpea':
            canvas.itemconfig(image_id, image=chickPeaPic)
        elif answerStr.get() == 'watermelon':
            canvas.itemconfig(image_id, image=watermelonPic)
        elif answerStr.get() == 'kidneybeans':
            canvas.itemconfig(image_id, image=beansPic)
        elif answerStr.get() == 'maize':
            canvas.itemconfig(image_id, image=maizePic)
        elif answerStr.get() == 'papaya':
            canvas.itemconfig(image_id, image=papayaPic)
        elif answerStr.get() == 'lentil':
            canvas.itemconfig(image_id, image=lentilPic)
        elif answerStr.get() == 'pomegranate':
            canvas.itemconfig(image_id, image=pomegranatePic)
        elif answerStr.get() == 'cotton':
            canvas.itemconfig(image_id, image=cottonPic)
        elif answerStr.get() == 'coconut':
            canvas.itemconfig(image_id, image=coconutPic)


# canvas for image
canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()
canvas.place(x=595, y=140)

# images
defaultImage = tk.PhotoImage(file="farmingLogo.png")
applePic = tk.PhotoImage(file="apple.png")
bananaPic = tk.PhotoImage(file="banana.png")
chickPeaPic = tk.PhotoImage(file="chickpea.png")
watermelonPic = tk.PhotoImage(file="watermelon.png")
muskMelonPic = tk.PhotoImage(file="muskmelon.png")
orangePic = tk.PhotoImage(file="orange.png")
coconutPic = tk.PhotoImage(file="coconut.png")
cottonPic = tk.PhotoImage(file="cotton.png")
grapesPic = tk.PhotoImage(file="grapes.png")
beansPic = tk.PhotoImage(file="kidneyBeans.png")
lentilPic = tk.PhotoImage(file="lentil.png")
maizePic = tk.PhotoImage(file="maize.png")
mangoPic = tk.PhotoImage(file="mango.png")
papayaPic = tk.PhotoImage(file="papaya.png")
pomegranatePic = tk.PhotoImage(file="pomegranate.png")

# set first image on canvas
image_id = canvas.create_image(0, 0, anchor='nw', image=defaultImage)


# This is the section of code which creates a button
Button(root, text='Farm!', bg='yellow', font=('arial', 20, 'bold'), command=farmBttnClicked ).place(x=188, y=403)

# This is the section of code which creates the a label
errorLabelstr = StringVar()
errorLabelstr.set("")
errorLabel = tk.Label(root, textvariable=errorLabelstr, bg='lemon chiffon', fg="red", font=('arial', 15, 'normal')).place(x=18, y=463)
    
                      
Label(root, text='ANSWERS', bg='lemon chiffon', font=('arial', 15, 'bold')).place(x=700, y=73)
answerStr = StringVar()
answerStr.set("")
Label(root, textvariable=answerStr, bg='tomato', font=('arial', 15, 'normal')).place(x=600, y=123)

root.mainloop()
