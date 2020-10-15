#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Chapter 1. 
"""
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

MAXVAL = 10
INTERVAL = (MAXVAL*10) + 1

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
t_xdata, t_ydata, h_xdata, h_ydata = [], [], [], [] 

def update():
    t_a = float(t_aSpbox.get())
    t_b = float(t_bSpbox.get())
    h_a = float(h_aSpbox.get())    
    doMeet = False

    f = open("01LinearFunction_log.txt", "a")
    for t in np.linspace(0, MAXVAL, INTERVAL):
        t_y = t_a*t + t_b
        h_y = h_a*t
        t_xdata.append(t)
        t_ydata.append(t_y)
        h_xdata.append(t)
        h_ydata.append(h_y)
        if(h_y >= t_y and (not doMeet)):
            doMeet = True
            meetTime = t
            meetDistance = t_y
        # write entry value to file

        f.write("x: "+str(math.ceil(t*100)/100)+", t_y: "+str(math.ceil(t_y*100)/100)+", h_y: "+str(math.ceil(h_y*100)/100)+"\n")
        # print("t : "+str(math.ceil(t*100)/100)+", t_y: "+str(math.ceil(t_y*100)/100)+", h_y : "+str(math.ceil(h_y*100)/100)+", doMeet : "+str(doMeet))

    f.close()
    ax.set_xlabel('Time(hour)')
    ax.set_ylabel('Distance(km)')
    ax.plot(t_xdata,t_ydata, label='Tortoise')
    ax.plot(h_xdata,h_ydata, label='Hare')

    if (doMeet):
        ax.set_title('The tortoise overcome from '+str(math.ceil(meetTime*100)/100)+'hour(s), '+str(math.ceil(meetDistance*100)/100)+'km(s)')
        ax.plot(meetTime, meetDistance, 'ro')
    else:
        ax.set_title('They will not meet')
    ax.legend()
    fig.canvas.draw()

#main
main = Tk()
main.title("The Hare and the Tortoise")
main.geometry()

label=Label(main, text='The Hare and the Tortoise')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

t_aVal  = DoubleVar(value=1.0)
t_bVal  = DoubleVar(value=4.0)
h_aVal  = DoubleVar(value=2.0)

t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=10, increment=1, justify=RIGHT)
t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='The tortoise (km/h) : ')                
t_aLabel.grid(row=1,column=0)

t_bSpbox = Spinbox(main, textvariable=t_bVal,from_=0, to=10, increment=1, justify=RIGHT)
t_bSpbox.config(state='readonly')
t_bSpbox.grid(row=2,column=1)
t_bLabel=Label(main, text='The tortoise (km) : ')                
t_bLabel.grid(row=2,column=0)

h_aSpbox = Spinbox(main, textvariable=h_aVal ,from_=0, to=10, increment=1, justify=RIGHT)
h_aSpbox.config(state='readonly')
h_aSpbox.grid(row=3,column=1)
h_aLabel=Label(main, text='The hare (km/h) : ')                
h_aLabel.grid(row=3,column=0)

Button(main,text="Run",width=20,height=5,command=lambda:update()).grid(row=1, column=2,columnspan=2, rowspan=3)

canvas = FigureCanvasTkAgg(fig, main)
canvas.get_tk_widget().grid(row=4,column=0,columnspan=3) 

# create initial blank file
f = open("01LinearFunction_log.txt", "w")
f.close()

main.mainloop()


# In[ ]:


"""
Chapter 1. 
"""
import numpy as np
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math

MAXVAL = 10
INTERVAL = MAXVAL + 1

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
h_xdata, h_ydata, x_list, y_list = [], [], [], []
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

grad_fig = Figure(figsize=(5,4), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(-0.1, 1.1)
grad_ax.set_ylim(0, 100)
t_xdata, t_ydata = [], []
ln, = grad_ax.plot(0, 0)
dn, = grad_ax.plot([], [], 'ro')

def update():
    h_a = int(h_aSpbox.get())    
    for t in np.linspace(0, MAXVAL, INTERVAL):
        h_y = h_a*t
        h_xdata.append(t)
        h_ydata.append(h_y)
    ax.set_xlabel('Time(hour)')
    ax.set_ylabel('Distance(km)')
    ax.set_title('Linear Regression')
    ax.plot(h_xdata,h_ydata, 'ro', label='Hare')
    ax.legend()
    fig.canvas.draw()

def get_cost(a_val):
    h_a = float(h_aSpbox.get()) 
    cost = 0
    for i in range(0, 11, 1):
        cost += pow((a_val*i - h_a*i),2)
    return cost

def showLines():
    h_a = float(h_aSpbox.get()) 
    h_s = float(h_sSpbox.get())  
    a_val = h_a + (h_s * 5) 
    h_xdata = []
    h_ydata = []
    for i in np.linspace(0, MAXVAL, INTERVAL):
        a = a_val - (i * h_s)
        for t in np.linspace(0, MAXVAL, INTERVAL):
            h_y = a*t
            h_xdata.append(t)
            h_ydata.append(h_y)
        ax.plot(h_xdata,h_ydata, alpha=0.2) 
    fig.canvas.draw()

def init():
    grad_ax.set_xlim(-0.1, 1.1)
    grad_ax.set_ylim(0, 100)
    return dn, ln, 

def animateFrame(frame):
    h_a = float(h_aSpbox.get()) 
    h_s = float(h_sSpbox.get())
    a_val = h_a + (h_s * 5) 
    i = frame * h_s
    a = a_val - i
    t_xdata.append(i)
    t_ydata.append(get_cost(a))
    dn.set_data(t_xdata, t_ydata)
    # ln.set_data(t_xdata, t_ydata)
    return dn, ln,

def gradient():
    ani = FuncAnimation(fig, animateFrame, frames=np.linspace(0, MAXVAL, INTERVAL),
                        init_func=init, blit=True)

    grad_ax.set_title('Gradient descent')
    grad_ax.set_ylabel("Total Cost")
    grad_ax.set_xlabel("Variance")

    grad_fig.canvas.draw()

#main
main = Tk()
main.title("The Hare Linear Regression")
main.geometry()

label=Label(main, text='The Hare Linear Regression')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

h_aVal  = DoubleVar(value=1.0)

h_aSpbox = Spinbox(main, textvariable=h_aVal ,from_=0, to=10, increment=1, justify=RIGHT)
h_aSpbox.config(state='readonly')
h_aSpbox.grid(row=1,column=2)
h_aLabel=Label(main, text='The hare (km/h) : ')                
h_aLabel.grid(row=1,column=0,columnspan=2)

h_sVal  = DoubleVar(value=0.1)

h_sSpbox = Spinbox(main, textvariable=h_sVal ,from_=0, to=2, increment=0.01, justify=RIGHT)
h_sSpbox.config(state='readonly')
h_sSpbox.grid(row=2,column=2)
h_sLabel=Label(main, text='Velocity variance (km/h) : ')                
h_sLabel.grid(row=2,column=0,columnspan=2)

Button(main,text="Run",width=20,height=3,command=lambda:update()).grid(row=3, column=0)
Button(main,text="Lines",width=20,height=3,command=lambda:showLines()).grid(row=3, column=1)
Button(main,text="Gradient",width=20,height=3,command=lambda:gradient()).grid(row=3, column=2)

canvas = FigureCanvasTkAgg(fig, main)
canvas.get_tk_widget().grid(row=4,column=0,columnspan=4)

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=5,column=0,columnspan=4) 

main.mainloop()


# In[ ]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.read_csv("./data/marathon_2015_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd.DataFrame(marathon_2015_2017,columns=['5K',  '10K',  '15K',  '20K', 'Half',  '25K',  '30K',  '35K',  '40K',  'Official Time']).sort_values(by=['Official Time'])

# Dataframe to List
record_list = record.values.tolist()

xData = [5, 10, 15, 20, 21.098, 25, 30, 35, 40, 42.195 ]

fig = Figure(figsize=(6,6), dpi=100)
ax = fig.add_subplot(111)
t_xdata, t_ydata, ml_xdata, ml_ydata, p_xdata, p_ydata = [], [], [], [], [], []
ax.set_xlim(0, 45)
ax.set_ylim(0, 13000)
ax.set_xlabel('Distance(km)')
ax.set_ylabel('Time(Second)')
ax.set_title('Records of runner')
ln, = ax.plot([], [], linestyle=':')
dn, = ax.plot([], [], 'ro')
pn, = ax.plot([], [], 'bs')
t_a = 0

grad_fig = Figure(figsize=(6,6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(0, 5000)
grad_ax.set_ylim(0, 50000)
grad_ax.set_title('Cost Gradient Decent')
grad_ax.set_ylabel("Total Cost")
grad_ax.set_xlabel("Number of Traning")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def init():
    t_a = int(t_aSpbox.get()) -1
    ax.set_title('Records of runner #'+str(t_a + 1))
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 13000)
    grad_ax.set_xlim(0, 5000)
    grad_ax.set_ylim(0, 50000)
    return dn,

def animateFrame(frame):
    t_a = int(t_aSpbox.get()) -1
    t_x = xData[int(frame)]
    t_y = record_list[t_a][int(frame)]
    t_xdata.append(t_x)
    t_ydata.append(t_y)  
    dn.set_data(t_xdata, t_ydata) 
    ax.annotate(seconds_to_hhmmss(t_y), (t_x, t_y), fontsize=8) 
    fig.canvas.draw()
    return dn,

def update(): 
    # Initialize t_xdata, t_ydata for ax graph
    t_xdata.clear()
    t_ydata.clear()
    
    ani = FuncAnimation(fig, animateFrame, frames=np.linspace(0, len(xData)-1, len(xData)),
                        init_func=init, blit=True, repeat = False)
    fig.canvas.draw()

def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_a = int(t_aSpbox.get())
    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())
        
    # X and Y data from 0km to 30km
    x_train = [ i/10 for i in xData[0:7]]
    y_train = record_list[t_a-1][0:7]
    
    # Try to find values for W and b to compute y_data = x_data * W + b
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")
    
    # placeholders for a tensor that will be always fed using feed_dict
    # See http://stackoverflow.com/questions/36693740/
    X = tf.placeholder(tf.float32, shape=[None])
    Y = tf.placeholder(tf.float32, shape=[None])
    
    # Our hypothesis XW+b
    hypothesis = X * W + b
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # optimizer
    train = tf.train.GradientDescentOptimizer(learning_rate=t_r).minimize(cost)
    
    # Launch the graph in a session.
    with tf.Session() as sess:
        # Initializes global variables 
        sess.run(tf.global_variables_initializer())
    
        # Fit the line
        log_ScrolledText.insert(END, "%10s %4i %10s %6i %20s %10.8f" % ('\nRunner #', t_a, ', No. of train is', (t_t-1), ', learing rate is ', t_r)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
        log_ScrolledText.insert(END, "%20s %20s %20s %20s" % ('Step', 'Cost', 'W', 'b')+'\n\n')
        for step in range(t_t):
            _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: x_train, Y: y_train})
    
            if step % 100 == 0:
                # print(step, cost_val, W_val, b_val) 
                g_xdata.append(step)
                g_ydata.append(cost_val)
                log_ScrolledText.insert(END, "%20i %20.5f %20.5f %20.5f" % (step, cost_val, W_val, b_val)+'\n')
        #gn.set_data(g_xdata, g_ydata)
        grad_ax.plot(g_xdata, g_ydata, 'ro')
        grad_ax.set_title('The minimum cost is '+str(cost_val)+' at '+str(step)+'times')
        grad_fig.canvas.draw()    
        
        # Testing our model
        log_ScrolledText.insert(END, "%20s" % ('\n\nHypothesis = X * W + b\n\n'), 'HEADER')
        draw_hypothesis(W_val, b_val)
        log_ScrolledText.insert(END, "%20s" % ('\n\nRecords Prediction\n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%20s %20s %20s %20s" % ('Distance(km)', 'Real record', 'ML Prediction', 'Variation(Second)')+'\n\n')
        for index in range(7, 10):
            x_value = xData[index] / 10
            p_xdata.append(xData[index])
            time = sess.run(hypothesis, feed_dict={X: [x_value]})
            p_ydata.append(time[0])
            log_ScrolledText.insert(END, "%20.3f %20s %20s %20i" % (xData[index], seconds_to_hhmmss(t_ydata[index]), seconds_to_hhmmss(time[0]), (t_ydata[index] - time[0]))+'\n')

        dn.set_data(t_xdata, t_ydata)  
        pn.set_data(p_xdata, p_ydata)
        fig.canvas.draw()        


def draw_hypothesis(W, b):
    # Clear line
    ml_xdata.clear()
    ml_ydata.clear()
    # Clear Prediction
    p_xdata.clear()
    p_ydata.clear()
        
    x_value = [ i/10 for i in xData]
    for x in range(10):
        #ax.annotate('', (t_xdata[i], t_ydata[i]), fontsize=8) 
        h = W * x_value[x] + b
        ml_xdata.append(xData[x])
        ml_ydata.append(h)
    ln.set_data(ml_xdata, ml_ydata)
    b_exp = ''
    if b > 0:
        b_exp = ' + '+str(b)
    elif b < 0:
        b_exp = ' - '+str(abs(b))        
    log_ScrolledText.insert(END, 'Hypothesis = X * '+str(W)+b_exp+'\n', 'RESULT')
    
#main
main = Tk()
main.title("Marathon Records")
main.geometry()

label=Label(main, text='Marathon Records Prediction by Machine Learing')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(record_list), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=5000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=0.01)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=0.001, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Get History", height=2,command=lambda:update()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Machine Learing", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

canvas = FigureCanvasTkAgg(fig, main)
canvas.get_tk_widget().grid(row=3,column=0,columnspan=3) 

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=3,columnspan=3)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd.DataFrame(marathon_2015_2017,columns=['M/F',  'Age',  'Pace',  '10K', '20K',  '30K',  'Official Time']).sort_values(by=['Official Time'])

record['M/F'] = record[____].map({'M': 1, ___: _})
# Dataframe to List
record_list = record.values.tolist()

grad_fig = Figure(figsize=(6,6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(0, 2000)
grad_ax.set_ylim(0, 10000)
grad_ax.set_title('Cost Gradient Decent')
grad_ax.set_ylabel("Total Cost")
grad_ax.set_xlabel("Number of Traning")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())
        
    # X and Y data from 0km to 30km
    
    x_train_1 = [ r[_] for r in record_list]
    x_train_2 = [ r[_] for r in record_list]
    x_train_3 = [ r[_] for r in record_list]

    y_train = [ r[__] for r in record_list]

    # Try to find values for W and b
    W1 = tf.Variable(tf.random_normal([1]), name="weight1")
    W2 = tf.Variable(tf.random_normal([1]), name="weight2")
    W3 = tf.Variable(tf.random_normal([1]), name="weight3")

    b = tf.Variable(tf.random_normal([1]), name="bias")
    
    # placeholders for a tensor
    X1 = tf.placeholder(tf.float32, shape=[____])
    X2 = tf.placeholder(tf.float32, shape=[____])
    X3 = tf.placeholder(tf.float32, shape=[____])

    Y = tf.placeholder(tf.float32, shape=[____])
    
    # Our hypothesis
    hypothesis = _____________________ + b
    
    # cost/loss function
    cost = tf.reduce_mean(tf.square(_________ - Y))
    
    # optimizer
    train = tf.train.GradientDescentOptimizer(__________=t_r).minimize(____)
    
    # Launch a session.
    with tf.______() as sess:
        # Initializes global variables
        sess.___(tf.______________________())
    
        # Fit the line
        log_ScrolledText.insert(END, "%10s %6i %20s %10.8f" % ('\nNo. of train is', (t_t-1), ', learing rate is ', t_r)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
        log_ScrolledText.insert(END, "%20s %20s" % ('Step', 'Cost')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val = sess.run([train, ____,  ________], feed_dict={X1: x_train_1, X2: x_train_2, X3: x_train_3, Y: y_train})
    
            if step % 100 == 0:
                print(step, cost_val, h_val) 
                g_xdata.append(step)
                g_ydata.append(cost_val)
                log_ScrolledText.insert(END, "%20i %20.5f" % (step, cost_val)+'\n')
        #gn.set_data(g_xdata, g_ydata)
        grad_ax.plot(g_xdata, g_ydata, 'ro')
        grad_ax.set_title('The minimum cost is '+str(cost_val)+' at '+str(step)+'times')
        grad_fig.canvas.draw() 
        # Testing our model
        winner = record_list[0]
        print(winner)
        time = sess.run(________, feed_dict={X1: [winner[_]], X2: [winner[_]], X3: [winner[_]]})
        # time = sess.run(hypothesis, feed_dict={X1: [1], X2: [25], X3: [296]})
        log_ScrolledText.insert(END, "%20s" % ('\n\nThe Winner Records Prediction\n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%20s %20s %20s" % ('Real record', 'ML Prediction', 'Variation(Second)')+'\n\n')
        log_ScrolledText.insert(END, "%20s %20s %20i" % (seconds_to_hhmmss(y_train[0]), seconds_to_hhmmss(time[_]), (y_train[_] - time[_]))+'\n')
        
    
#main
main = Tk()
main.title("Marathon Records")
main.geometry()

label=Label(main, text='Multi Variable Linear Regression Concept')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

t_tVal  = IntVar(value=2000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=1)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=0)

t_rVal  = DoubleVar(value=1e-6)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-6, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=3)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=2)

Button(main,text="Machine Learing", height=2,command=lambda:learing()).grid(row=2, column=0, columnspan=4, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=4)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=4, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:


import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd.DataFrame(marathon_2015_2017,columns=['M/F',  'Age',  'Pace',  '10K', '20K',  '30K',  'Official Time']).sort_values(by=['Official Time'])

record['M/F'] = record[____].map({'M': 1, ___: _})
# Dataframe to List
record_list = record.values.tolist()

gender_list = ['Female', 'Male']
grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def histogram():
    gender = t_gCbbox.get()
    t_g = int(gender_list.index(gender)) 
    t_a = int(t_aSpbox.get())
    t_p = int(t_pSpbox.get())
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'  
    gender_record = record[record['M/F'] == t_g]
    gender_age_record = gender_record[gender_record.Age == t_a-1] 
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5)
    grad_ax.plot(t_a, t_p, 'yd')
    stat = gender_age_record['Pace'].describe()
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_a)
    grad_ax.set_title(title)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1050), fontsize=10)
        
    grad_fig.canvas.draw()     

def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import _________ as tf
    tf.set_random_seed(777)  # for reproducibility

    gender = t_gCbbox.get()
    t_g = int(gender_list.index(gender))    
    t_a = int(t_aSpbox.get()) 
    t_p = int(t_pSpbox.get())

    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())
        
        
    # X and Y data from 0km to 30km 
    x_train = [ r[___] for r in record_list ]
    y_train = [ [r[__]] for r in record_list ]

    # Try to find values 
    W = tf.Variable(tf.random_normal([_, _]), name='weight')
    b = tf.Variable(tf.random_normal([_]), name="bias")
    
    # placeholders for a tensor 
    X = tf.placeholder(tf.float32, shape=[None, _])
    Y = tf.placeholder(tf.float32, shape=[None, _])
    
    # Our hypothesis 
    # hypothesis = X1 * W1 + X2 * W2 + X3 * W3 + b
    hypothesis = tf.______(_, _) + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(________ - Y))
    
    # optimizer
    train = tf.train.GradientDescentOptimizer(___________=t_r).minimize(____)
    
    # Launch a session.
    with tf._______() as sess:
        # Initializes global variables 
        sess.___(tf._____________________())

        # Fit the line
        #log_ScrolledText.insert(END, "%10s %6s %10s %3s %10s %5s" % ('\nGender :', gender_list[t_g], ', Age :', t_a, ', Pace :'+ t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_a)+', Pace :'+str(t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
        log_ScrolledText.insert(END, "%20s %20s" % ('Step', 'Cost')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val = sess.run([train, cost,  __________], feed_dict={X: ________, Y: y_train})
    
            if step % 100 == 0:
                print(step, cost_val, h_val[0]) 
                log_ScrolledText.insert(END, "%20i %20.5f" % (step, cost_val)+'\n')

        # Testing our model
        winner = [ t_g, t_a, t_p ]
        time = sess.___(___________, feed_dict={X: [________]})
        ml_time = seconds_to_hhmmss(time[0][0]) + '(' + str(time[0][0]) + ')'
        # time = sess.run(hypothesis, feed_dict={X1: [1], X2: [25], X3: [296]})
        log_ScrolledText.insert(END, "%20s" % ('\n\nThe Prediction Records\n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%10s %10s %10s %50s" % ('Gender', 'Age', 'Pace','Record Prediction(Second) at 42.195km')+'\n\n')
        log_ScrolledText.insert(END, "%10s %10s %10s %50s" % (gender_list[t_g], str(t_a), str(t_p), ml_time)+'\n') 
            
#main
main = Tk()
main.title("Multi Variable Matrix Linear Regression")
main.geometry()

label=Label(main, text='Multi Variable Matrix Linear Regression')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_gVal  = StringVar(value=gender_list[0])
t_gCbbox = ttk.Combobox(main, textvariable=t_gVal)
t_gCbbox['values'] = gender_list
t_gCbbox.config(state='readonly')
t_gCbbox.grid(row=1,column=1)

t_gLabel=Label(main, text='Gender : ')                
t_gLabel.grid(row=1,column=0)

t_aVal  = IntVar(value=45)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=18, to=84, increment=1, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=3)
t_aLabel=Label(main, text='Age : ')                
t_aLabel.grid(row=1,column=2)

t_pVal  = IntVar(value=500)
t_pSpbox = Spinbox(main, textvariable=t_pVal ,from_=0, to=1500, increment=1, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_pSpbox.grid(row=1,column=5)
t_pLabel=Label(main, text='Pace : ')                
t_pLabel.grid(row=1,column=4)


t_tVal  = IntVar(value=2000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=2,column=1)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=2,column=0)

t_rVal  = DoubleVar(value=1e-6)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-6, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=2,column=3)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=2,column=2)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=4, columnspan=1, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=5, columnspan=1, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.________("./data/marathon_2015_2017.csv")
marathon_2015_2017['M/F'] = marathon_2015_2017[____].map({'M': 1, ___: _})

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
marathon_2015_2016 = marathon_2015_2017[marathon_2015_2017['Year'] != 2017]
marathon_2017 = marathon_2015_2017[marathon_2015_2017['Year'] == 2017]

df_2015_2016 = pd.DataFrame(marathon_2015_2016,columns=['M/F',  'Age',  'Pace',  '10K', '20K',  '30K',  'Official Time'])
df_2017 = pd.DataFrame(marathon_2017,columns=['M/F',  'Age',  'Pace',  '10K', '20K',  '30K',  'Official Time'])

# Dataframe to List
record_2015_2016 = df_2015_2016.values.tolist()
record_2017 = df_2017.values.tolist()

# X and Y data    
x_train = [ r[___] for r in record_2015_2016]
y_train = [ r[___] for r in record_2015_2016]

x_test = [ r[___] for r in record_2017]
y_test = [ r[___] for r in record_2017]

gender_list = ['Female', 'Male']
grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")
g_xdata, g_ydata = [], []
gn, = grad_ax.plot([], [], 'ro')

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def histogram():
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    print(runner)
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'  
    gender_record = df_2017[df_2017['M/F'] == t_g]
    gender_age_record = gender_record[gender_record.Age == t_y] 
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5)
    grad_ax.plot(t_y, t_p, 'yd')
    stat = gender_age_record['Pace'].________()
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_y)+', Pace : '+str(t_p)
    grad_ax.set_title(title)
    grad_ax.annotate('['+gender_list[t_g]+', '+str(t_y)+']', (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1050), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1000), fontsize=10)
        
    grad_fig.canvas.draw()    
    
def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])

    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())

    # placeholders for a tensor that will be always fed.
    W = tf.Variable(tf.random_normal([_, _]), name='weight')
    # Same to the number of output
    b = tf.Variable(tf.random_normal([_]), name='bias')
    
    X = tf.placeholder(tf.float32, shape=[None, _])
    Y = tf.placeholder(tf.float32, shape=[None, _])

    # Hypothesis
    hypothesis = tf.matmul(X, W) + b

    # Simplified cost/loss function
    cost = tf.reduce_mean(tf.square(_________ - Y))
    
    # optimizer
    train = tf.train.GradientDescentOptimizer(____________=t_r).minimize(____)
    
    # Launch a session.
    with tf.Session() as sess:
        # Initializes global variables 
        sess.___(tf._______________________())
    
        # Fit the line
        log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_y)+', Pace :'+str(t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
        log_ScrolledText.insert(END, "%10s %20s %50s" % ('Step', 'Cost', 'Hypothesis')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val = sess.run([train, cost, hypothesis], feed_dict={X: x_train, Y: y_train})
            
            if step % 100 == 0:
                print(step, cost_val, h_val[t_a]) 
                log_ScrolledText.insert(END, "%10i %20.5f %50s" % (step, cost_val, h_val[t_a])+'\n')

        # Testing our model
        winner = [ t_g, t_y, t_p ]
        time = sess.run(_________, feed_dict={X: [winner]})

        #variation = y_test[0][0] - time[0]
        log_ScrolledText.insert(END, "%20s" % ('\n\nRecords Prediction\n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%20s %30s %30s %20s" % ('Distance(km)', 'Real record', 'ML Prediction', 'Variation(Second)')+'\n\n')
        distance = [ 10., 20., 30., 42.195 ]
        for i in range(len(time[0])):
            real_time = seconds_to_hhmmss(y_test[t_a][i]) + '(' + str(y_test[t_a][i]) + ')'
            ml_time = seconds_to_hhmmss(time[0][i]) + '(' + str(time[0][i]) + ')'
            variation = y_test[t_a][i] - time[0][i]

            log_ScrolledText.insert(END, "%20.3f %30s %30s %20.3f" % (distance[i], real_time, ml_time, variation)+'\n')         
    
#main
main = Tk()
main.title("Multi Variable Output Linear Regression")
main.geometry()

label=Label(main, text='Multi Variable Output Linear Regression')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(x_test), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=2000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=1e-6)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-6, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('RESULT', foreground='blue', font=("Helvetica", 12))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()



# In[ ]:


# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_qualifying_time.csv" under "data" folder
marathon_qualifying_time = pd.read_csv("./data/marathon_qualifying_time.csv")

# Drop unnecessary columns 
qualifying_time = pd.DataFrame(marathon_qualifying_time,columns=['F',  'M'])
# Import Numpy Library and call it as np
import numpy as np

# Convert using pandas to_timedelta method
qualifying_time['F'] = pd.to_timedelta(qualifying_time['F'])
qualifying_time['M'] = pd.to_timedelta(qualifying_time['M'])

# Convert time to seconds value using astype method
qualifying_time['F'] = qualifying_time['F'].astype('m8[s]').astype(np.int64)
qualifying_time['M'] = qualifying_time['M'].astype('m8[s]').astype(np.int64)

# Load the CSV
# files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")
marathon_2015_2017['M/F'] = marathon_2015_2017['M/F'].map({'M': 1, ___: _})

# Add qualifying column with fixed string 2017
qualifying_time_list = qualifying_time.values.tolist()
# Define function name to_seconds
marathon_2015_2017_qualifying = pd.DataFrame(columns=['M/F',  'Age',  'Pace',  'Official Time', 'Year', 'qualifying'])
for index, record in marathon_2015_2017.__________:
    qualifying_standard = qualifying_time_list[record.Age-__][record['M/F']]
    qualifying_status = 1
    if (record['Official Time'] > ______________): 
        qualifying_status = 0
    marathon_2015_2017_qualifying = marathon_2015_2017_qualifying.append({'M/F' : record['M/F'],
                                                                          'Age' : record['Age'],
                                                                          'Pace' : record['Pace'],
                                                                          'Official Time' : record['Official Time'],
                                                                          'Year' : record['Year'],
                                                                          'qualifying' : _____________
                                                                          },
                                                                        ignore_index=True)
# Save to CSV file "marathon_2015_2017.csv"
marathon_2015_2017_qualifying.______("./data/marathon_2015_2017_qualifying.csv", index = None, header=True)


# In[ ]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017_qualifying = pd._________("./data/marathon_2015_2017_qualifying.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
marathon_2015_2016 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]
marathon_2017 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]

df_2015_2016 = pd.DataFrame(marathon_2015_2016,______=['M/F',  'Age',  'Pace',  'qualifying'])
df_2017 = pd.DataFrame(marathon_2017,_______=['M/F',  'Age',  'Pace',  'qualifying'])

# Dataframe to List
record_2015_2016 = df_2015_2016.values.tolist()
record_2017 = df_2017.values.tolist()

gender_list = ['Female', 'Male']
grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def normalization(record):
    r0 = record[0]
    r1 = record[1] / 10
    r2 = record[2] / 100
    return [r0, r1, r2]

# X and Y data from 0km to 30km    
# x_train = [ r[0:3] for r in record_2015_2016]
x_train = [ normalization(r[0:3]) for r in record_2015_2016]
y_train = [ [r[-1]] for r in record_2015_2016]
# x_test = [ r[0:3] for r in record_2017]
x_test = [ r[0:3] for r in record_2017]
y_test = [ [r[-1]] for r in record_2017]

def histogram():
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    print(runner)
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'  
    gender_record = df_2017[df_2017['M/F'] == ___]
    gender_age_record = gender_record[gender_record.Age == ___] 
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5)
    grad_ax.plot(t_y, t_p, 'yd')
    stat = gender_age_record['Pace']._________
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_y)+', Pace : '+str(t_p)
    grad_ax.set_title(title)
    grad_ax.annotate('['+gender_list[t_g]+', '+str(t_y)+']', (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1050), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1000), fontsize=10)
        
    grad_fig.canvas.draw()    
    
def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])

    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())

    # placeholders for a tensor 
    W = tf.Variable(tf.random_normal([____]), name='weight')
    b = tf.Variable(tf.random_normal([_]), name='bias')
    
    X = tf.placeholder(tf.float32, shape=[None, _])
    Y = tf.placeholder(tf.float32, shape=[None, _])

    # Hypothesis
    hypothesis = tf.______(tf.matmul(X, W) + b)

    # Simplified cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    
    # optimizer
    train = tf.train.____________________(learning_rate=___).minimize(____)
    
    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(_________ > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(________, Y), dtype=tf.float32))

    # Launch a session.
    with tf.Session() as sess:
        # Initializes global variables 
        sess.run(tf.______________________)
    
        # Fit the line
        log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_y)+', Pace :'+str(t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent & Accuracy\n\n','HEADER')
        log_ScrolledText.insert(END, "%10s %20s %20s" % ('Step', 'Cost', 'Accuracy(%)')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val, p_val, a_val = sess.run([train, cost, _________, _________, ________], feed_dict={X: x_train, Y: y_train})
            
            if step % 100 == 0:
                print(step, cost_val, a_val) 
                log_ScrolledText.insert(END, "%10i %20.5f %20.7f" % (step, cost_val, a_val*100)+'\n')
        
        winner = [ t_g, t_y, t_p ]
        result = sess.run(hypothesis, feed_dict={X: [normalization(winner)]})
        log_ScrolledText.insert(END, '\n\n')
        log_ScrolledText.insert(END, "%10s %20s" % ('Value        ', 'Qualifying Prediction\n\n'), 'HEADER')
        if(result[0] > 0.5):
            log_ScrolledText.insert(END, "%10.7f %20s" % (result[0], 'Qualifier\n\n'), 'Qualifier')
        else:
            log_ScrolledText.insert(END, "%10.7f %20s" % (result[0], 'DisQualifier\n\n'), 'DisQualifier')        
    
#main
main = Tk()
main.title("Logistic Regression Binary Classification")
main.geometry()

label=Label(main, text='Logistic Regression Binary Classification')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(x_test), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=10000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=1e-2)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-2, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('Qualifier', foreground='blue', font=("Helvetica", 16))
log_ScrolledText.tag_config('DisQualifier', foreground='red', font=("Helvetica", 16))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:


import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017_qualifying = pd.read_csv("./data/marathon_2015_2017_qualifying.csv")
marathon_2015_2017_qualifying["Grade"] = 1
statistics_2015_2017 = marathon_2015_2017_qualifying["Official Time"]._________

marathon_2015_2017_qualifying.loc[marathon_2015_2017_qualifying["Official Time"] < statistics_2015_2017["25%"], "Grade"] = 0
marathon_2015_2017_qualifying.loc[marathon_2015_2017_qualifying["Official Time"] > statistics_2015_2017["75%"], "Grade"] = 2
'''
count    79638.000000
mean     13989.929167
std       2492.272069
min       7757.000000
25%      12258.000000
50%      13592.000000
75%      15325.000000
max      37823.000000
Name: Official Time, dtype: float64
'''
marathon_2015_2016 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]
marathon_2017 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]

df_2015_2016 = pd.DataFrame(marathon_2015_2016,columns=['M/F',  'Age',  'Pace',  'Grade'])
df_2017 = pd.DataFrame(marathon_2017,columns=['M/F',  'Age',  'Pace',  'Grade'])

# Dataframe to List
record_2015_2016 = df_2015_2016.values.tolist()
record_2017 = df_2017.values.tolist()

nb_classes = 3  # 0 ~ 2
gender_list = ['Female', 'Male']
grade_list = ['Outstanding(>25%)', 'Average(25~75%)', 'Below(<75%)']

grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def normalization(record):
    r0 = record[0]
    r1 = record[1] / 10
    r2 = record[2] / 100
    return [r0, r1, r2]

# X and Y data 
x_train = [ normalization(r[___]) for r in record_2015_2016]
y_train = [ [r[__]] for r in record_2015_2016]
# x_test = [ r[0:3] for r in record_2017]
x_test = [ r[___] for r in record_2017]
y_test = [ [r[__]] for r in record_2017]

def histogram():
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    print(runner)
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'  
    gender_record = df_2017[df_2017['M/F'] == t_g]
    gender_age_record = gender_record[gender_record.Age == t_y] 
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5)
    grad_ax.plot(t_y, t_p, 'yd')
    stat = gender_age_record['Pace'].describe()
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_y)+', Pace : '+str(t_p)
    grad_ax.set_title(title)
    grad_ax.annotate('['+gender_list[t_g]+', '+str(t_y)+']', (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1050), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1000), fontsize=10)
        
    grad_fig.canvas.draw()    
    
def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow 
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])

    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())

    # placeholders for a tensor 
    W = tf.Variable(tf.random_normal([__________]), name='weight')
    b = tf.Variable(tf.random_normal([__________]), name='bias')
    
    X = tf.placeholder(tf.float32, shape=[None, _])
    Y = tf.placeholder(tf.int32, shape=[None, _])

    Y_one_hot = tf.one_hot(__________)  # one hot
    print("one_hot:", Y_one_hot)
    Y_one_hot = tf.________(Y_one_hot, [-1, nb_classes])
    print("reshape one_hot:", Y_one_hot)

    '''
    one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
    reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
    '''
    # tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.______(logits)

    # Cross entropy cost/loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_____,
                                                                     labels=tf.stop_gradient([________])))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    prediction = tf.______(hypothesis, 1)
    correct_prediction = tf.equal(_________, tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(________________, tf.float32))

    # Launch a session.
    with tf.Session() as sess:
        # Initializes global variables
        sess.run(tf.global_variables_initializer())
    
        # Fit the line
        log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_y)+', Pace :'+str(t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent & Accuracy\n\n','HEADER')
        log_ScrolledText.insert(END, "%10s %20s %20s" % ('Step', 'Cost', 'Accuracy(%)')+'\n\n')
        for step in range(t_t):
            _, cost_val, a_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_train, Y: y_train})
            
            if step % 100 == 0:
                print("Step: {:5}\tCost: {:.3f}\tAccuracy: {:.2%}".format(step, cost_val, a_val))
                log_ScrolledText.insert(END, "%10i %20.5f %20.7f" % (step, cost_val, a_val*100)+'\n')
        
        winner = [ t_g, t_y, t_p ]
        result = sess.run(hypothesis, feed_dict={X: [normalization(winner)]})
        grade_index = sess.run(tf.argmax(result, axis=1))
        grade = grade_list[grade_index[0]]
        log_ScrolledText.insert(END, '\n\n')
        log_ScrolledText.insert(END, "%30s" % ('One Hot & Grade Prediction  \n\n'), 'HEADER')
        log_ScrolledText.insert(END, "%30s" % (result))             
        log_ScrolledText.insert(END, '\n\n')
        if(grade_index[0]):
            log_ScrolledText.insert(END, "%30s" % (grade+'\n\n'), 'DisQualifier')
        else:
            log_ScrolledText.insert(END, "%30s" % (grade+'\n\n'), 'Qualifier') 
            
#main
main = Tk()
main.title("Logistic Regression Multinominal Classification")
main.geometry()

label=Label(main, text='Logistic Regression Multinominal Classification')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(x_test), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=10000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=1e-2)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-2, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('Qualifier', foreground='blue', font=("Helvetica", 16))
log_ScrolledText.tag_config('DisQualifier', foreground='red', font=("Helvetica", 16))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()


# In[ ]:




