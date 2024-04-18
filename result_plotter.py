import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:/Users/Lenovo/Documents/Master/result_3d.csv"

dat = pd.read_csv(path,delimiter=',', header=0)

method_list = ['double_shade_pso','double_shade_bat','shade','random_search','double_shade','jderpo', 'jde']
func_list   = ['Rotated_Hyper_Ellipsoid','Himmelblau','Rosenbrock',   'Hartman_3D', 'Rastrigin', 'Levy'] #'Eggholder', 'Ackley', 'Michalewicz', 
nfe_list    = ['1e5', '2e5', '5e5']

#Finn max random search val
max_dict = {}
correct = False
for index, row in dat.iterrows():     
    if row[1] == 'random_search' and row[5] == 5e5:
        max_dict[row[3]] = {'score_in_threshold','score on contour'}
        max_dict[row[3]] = {'score_in_threshold','score on contour'}
        function = row[3]
        correct = True
    
    if row[0] == 'score within threshold' and correct:
        max_dict[function, 'score within threshold'] = row[1]
        score_in_threshold  = float(row[1])
    elif row[0] == 'score on contour' and correct:
        max_dict[function, 'score on contour'] = row[1]
        score_on_contour    = float(row[1])
        correct = False
    

score_in_threshold  = 'nan'
score_on_contour    = 'nan'
minima              = 'nan'
contour_            = 'nan'
below_thresh_       = 'nan'
count = 0

marker_styles = ['o', 's', '^', 'v', 'D', 'x', '+']
def contour_plot():
    method = method_list[0]; function = func_list[0]; nfe = nfe_list[0]
    fig, axs = plt.subplots(3,2, figsize = (10,10), sharey=True, sharex=True)
    lab_list = []
    for index, row in dat.iterrows():     
        if row[0] == 'Method:':
            method = 'nan'; function = 'nan'; nfe = 'nan'
            method  = row[1]
            function= row[3]
            nfe     = row[5]
            if nfe == 1e5:
                nfe = '1e5'
            elif nfe == 2e5:
                nfe = '2e5'
            elif nfe == 5e5:
                nfe = '5e5'

        if method == 'jde':
            mark = marker_styles[0]
        if method == 'shade':
            mark = marker_styles[1]
        if method == 'random_search':
            mark = marker_styles[2]
        if method == 'double_shade':
            mark = marker_styles[3]
        if method == 'double_shade_pso':
            mark = marker_styles[4]
        if method == 'double_shade_bat':
            mark = marker_styles[5]
        if method == 'jderpo':
            mark = marker_styles[6]
            
        if row[0] == 'score within threshold':
            score_in_threshold  = float(row[1])
        if row[0] == 'score on contour':
            score_on_contour    = float(row[1])
        if row[0] == 'min':
            minima              = float(row[1])
        if row[0] == 'contour %':
            contour_            = float(row[1])
        if row[0] == 'below threshold %':
            below_thresh_       = float(row[1])
            #Alt er hentet inn! Plottetid
            norm = float(max_dict[function,'score on contour'])
            if function == func_list[0]:
                lab = method + nfe
                lab_list.append(lab)
                axs[0,0].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[0,0].set_xlim(-0.2,1.5)
                axs[0,0].set_ylim(0,1.1)
                axs[0,0].set_title(function)
                axs[0,0].set_ylabel('contour fill %')
            if function == func_list[1]:
                lab = method + nfe
                axs[1,0].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[1,0].set_xlim(-0.2,1.5)
                axs[1,0].set_ylim(0,1.1)
                axs[1,0].set_title(function)
                axs[1,0].set_ylabel('contour fill %')
            if function == func_list[2]:
                lab = method + nfe
                axs[2,0].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[2,0].set_xlim(-0.2,1.5)
                axs[2,0].set_ylim(0,1.1)
                axs[2,0].set_title(function)
                axs[2,0].set_xlabel('1 - score on contour/worst score')
                axs[2,0].set_ylabel('contour fill %')
            if function == func_list[3]:
                lab = method + nfe
                axs[0,1].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[0,1].set_xlim(-0.2, 1.5)
                axs[0,1].set_ylim(0,1.1)
                axs[0,1].set_title(function)
            if function == func_list[4]:
                lab = method + nfe
                axs[1,1].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[1,1].set_xlim(-0.2,1.5)
                axs[1,1].set_ylim(0,1.1)
                axs[1,1].set_title(function)
            if function == func_list[5]:
                lab = method + nfe
                axs[2,1].scatter(1-score_on_contour/norm, contour_, label = lab, marker = mark)
                axs[2,1].set_xlim(-0.2,1.5)
                axs[2,1].set_ylim(0,1.1)
                axs[2,1].set_title(function)
                axs[2,1].set_xlabel('1 - score on contour/worst score')
    fig.subplots_adjust(right=0.7)
    fig.legend(lab_list, loc = 'upper right', bbox_to_anchor=(0.95, 0.8))
    plt.show()
    
def threshold_plot():
    
    fig, axs = plt.subplots(3,2, figsize = (10,10), sharey=True, sharex=True)
    lab_list = []
    method = method_list[0]; function = func_list[0]; nfe = nfe_list[0]
    for index, row in dat.iterrows():     

        if row[0] == 'Method:':
            method = 'nan'; function = 'nan'; nfe = 'nan'
            method  = row[1]
            function= row[3]
            nfe     = row[5]
            if nfe == 1e5:
                nfe = '1e5'
            elif nfe == 2e5:
                nfe = '2e5'
            elif nfe == 5e5:
                nfe = '5e5'

        if method == 'jde':
            mark = marker_styles[0]
        if method == 'shade':
            mark = marker_styles[1]
        if method == 'random_search':
            mark = marker_styles[2]
        if method == 'double_shade':
            mark = marker_styles[3]
        if method == 'double_shade_pso':
            mark = marker_styles[4]
        if method == 'double_shade_bat':
            mark = marker_styles[5]
        if method == 'jderpo':
            mark = marker_styles[6]
            
        if row[0] == 'score within threshold':
            score_in_threshold  = float(row[1])
        if row[0] == 'score on contour':
            score_on_contour    = float(row[1])
        if row[0] == 'min':
            minima              = float(row[1])
        if row[0] == 'contour %':
            contour_            = float(row[1])
        if row[0] == 'below threshold %':
            below_thresh_       = float(row[1])
            #Alt er hentet inn! Plottetid
            norm = float(max_dict[function,'score within threshold'])
            if function == func_list[0]:
                lab = method + nfe
                lab_list.append(lab)
                axs[0,0].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[0,0].set_xlim(-0.2,1.5)
                axs[0,0].set_ylim(0,1.1)
                axs[0,0].set_title(function)
                axs[0,0].set_ylabel('contour fill %')
            if function == func_list[1]:
                lab = method + nfe
                axs[1,0].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[1,0].set_xlim(-0.2,1.5)
                axs[1,0].set_ylim(0,1.1)
                axs[1,0].set_title(function)
                axs[1,0].set_ylabel('contour fill %')
            if function == func_list[2]:
                lab = method + nfe
                axs[2,0].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[2,0].set_xlim(-0.2,1.5)
                axs[2,0].set_ylim(0,1.1)
                axs[2,0].set_title(function)
                axs[2,0].set_xlabel('1 - score in threshold/worst score')
                axs[2,0].set_ylabel('contour fill %')
            if function == func_list[3]:
                lab = method + nfe
                axs[0,1].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[0,1].set_xlim(-0.2, 1.5)
                axs[0,1].set_ylim(0,1.1)
                axs[0,1].set_title(function)
            if function == func_list[4]:
                lab = method + nfe
                axs[1,1].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[1,1].set_xlim(-0.2,1.5)
                axs[1,1].set_ylim(0,1.1)
                axs[1,1].set_title(function)
            if function == func_list[5]:
                lab = method + nfe
                axs[2,1].scatter(1-score_in_threshold/norm, below_thresh_, label = lab, marker = mark)
                axs[2,1].set_xlim(-0.2,1.5)
                axs[2,1].set_ylim(0,1.1)
                axs[2,1].set_title(function)
                axs[2,1].set_xlabel('1 - score in_threshold/worst score')
    fig.subplots_adjust(right=0.7)
    fig.legend(lab_list, loc = 'upper right', bbox_to_anchor=(0.95, 0.8))
    plt.show()
    
threshold_plot()
