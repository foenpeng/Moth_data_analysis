import os
import csv
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from scipy import stats
from scipy import integrate

accel_vol = 9.8/12 #9.8 kg/s^2 per 12 voltage value changes
sample_freq = 1000 # Hz
PeakThreshold = 3*9.8 # the threshold for peak detection
Dist = 10 #the minnimum distance between two peaks 

#morph = "a+001c-001l070r1.5R025v020p000"
morph = str(input("Which morphology is it?\n")) + "l070r1.5R025v020p000"
morph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),morph)
trial_dir = os.listdir(morph_path)

visit_info = []
trial_count = 0
for item in trial_dir:
    trial_path = os.path.join(morph_path, item)
    if os.path.isdir(trial_path):
        trial_count += 1
        trial_folder = os.path.basename(trial_path)
        print(trial_folder)
        os.chdir(trial_path)
        e_file = open('e_data.csv', 'r+')
        e_read =  csv.reader(e_file)
        m_file = open('m_data.csv', 'r')
        m_read = csv.reader(m_file)
        x_file = open('x_data.csv', 'r')
        x_read = csv.reader(x_file)
        y_file = open('y_data.csv', 'r')
        y_read = csv.reader(y_file)
        z_file = open('z_data.csv', 'r')
        z_read = csv.reader(z_file)
        visit_file = open('visit_data.csv', 'w')
        visit_write = csv.writer(visit_file, delimiter=',', lineterminator='\n')
        
        # find all the nectar empty events
        empty_time = []
        fill_time = []
        e_data=[]
        for row in e_read:
            e_data.append([int(row[0]),float(row[1])])
        
        # check if  e_data is correct. 
        for i in range(1,len(e_data)-1):
            if e_data[i+1][1]-e_data[i][1]<0.5 or e_data[i][1]-e_data[i-1][1]<0.5:
                pass
            elif e_data[i][0] == 0:
                empty_time.append(round(float(e_data[i][1]),2))
                
        # read moth present/absent data, and x, y, z data
        moth = []
        for row in m_read:
            moth.append([float(row[0]), float(row[1])])
        moth = np.array(moth)
        
        x = []
        for row in x_read:
            x.append([float(row[0]), float(row[1])])
        x = np.array(x)
        t = x[:,1]
        x = x[:,0]
        x = (x - stats.mode(x)[0])*accel_vol

        y = []
        for row in y_read:
            y.append([float(row[0]), float(row[1])])
        y = np.array(y)[:,0]
        y = (y - stats.mode(y)[0])*accel_vol
        
        z = []
        for row in z_read:
            z.append([float(row[0]), float(row[1])])
        z = np.array(z)[:,0]
        z = (z - stats.mode(z)[0])*accel_vol
        

        accel_raw = np.vstack((x, y, z))
       #accel= np.sqrt((np.power(accel_raw[0,:],2) + np.power(accel_raw[1,:],2) + np.power(accel_raw[2,:],2))) 
        #hit_count1 = peakutils.indexes(accel, thres=PeakThreshold /max(accel), min_dist=10)

        # read the comment file
        with open("comments.txt") as comment_file:
            comments = comment_file.readlines()
        
        moth_sex = comments[13].rstrip('\n')
        moth_weight = comments[15].rstrip('\n')
        prob_length = comments[19].rstrip('\n')
        eclosion_days =  comments[21].rstrip('\n')
        tem = comments[9].rstrip('\n')
        hum = comments[11].rstrip('\n')
        
        # find the indice of consecutive moth absence time. delay defines how much time moth absence, 2 = 0.4s
        def find_moth_absence(data, delay):
            zeros = np.where(data==0)[0]
            zeros_split = np.split(zeros, np.where(np.diff(zeros) != 1)[0]+1)
            zerorun= [array for array in zeros_split if len(array) > delay]
            return np.concatenate(zerorun)
           
        zeros = find_moth_absence(moth[:,0],4)

        
        # define visits based on moth present before and after nectar empty event
        visit_time = []
        for index, time_stamp in enumerate(empty_time):
            idx = (np.abs(moth[:,1] - time_stamp)).argmin()
            point = (np.abs(zeros - idx)).argmin()        
            if idx < zeros[point]:
                visit_start = moth[zeros[point-1]+1,1]
                visit_end = moth[zeros[point]-1,1]
            else:
                visit_start = moth[zeros[point]+1,1]
                visit_end = moth[zeros[point+1]-1,1]
            if visit_start < time_stamp < visit_end:
                visit_time.append([round(visit_start,2), time_stamp, round(visit_end,2)])
                # check if there is nectar level fluctuation during one single visit
                if len(visit_time) > 1:
                    if visit_time[-1][0] == visit_time[-2][0]:
                        print("Nectar sensing fluctuation occurs at {0}".format(time_stamp))
                        del visit_time[-2]
            else:
                print("Reading start and end time error occurs in {0}".format(time_stamp))
        
        
        visit_num =  len(visit_time)
        visit_write.writerow(["trial_name", "trial_number", "visit_number","days_after_eclosion", 'moth_sex','moth_weight','prob_length','tem', 'hum', 'visit_start', 'nectar_empty', 'visit_end', 'start_empty', 'start_end', 'hit_count'])
        visit_count = 0
        for visit in visit_time:
            visit_count += 1
            xyz_start = (np.abs(t - visit[0])).argmin()
            xyz_end  = (np.abs(t- visit[2])).argmin()
            xyz_index = range(xyz_start,xyz_end)
            t_visit = t[xyz_index]
         
            accel_visit = np.sqrt((np.power(accel_raw[0,xyz_index],2) + np.power(accel_raw[1,xyz_index],2) + np.power(accel_raw[2,xyz_index],2))) 
            hit_count = peakutils.indexes(accel_visit, thres=PeakThreshold /max(accel_visit), min_dist=Dist)
            #pplot(t_visit,accel_visit,hit_count)
            
            this_visit = [trial_folder, trial_count, visit_count,  eclosion_days, moth_sex,moth_weight,prob_length,tem, hum,  visit[0],visit[1],visit[2], visit[1]-visit[0], visit[2]-visit[0], len(hit_count)]
            visit_info.append(this_visit)
            visit_write.writerow(this_visit)

        array = np.array(visit_info)
        e_file.close()
        m_file.close()
        x_file.close()
        y_file.close()
        z_file.close()
        visit_file.close()
        
        #pplot(t,accel,hit_count1)
        #plt.show()
        #break

os.chdir(morph_path)
with open(("{0}.csv").format(morph), "w") as morph_file:
    morph_write = csv.writer(morph_file,delimiter=',', lineterminator='\n')
    morph_write.writerow(["trial_name", "trial_number", "visit_number", "days_after_eclosion",'moth_sex','moth_weight','prob_length','tem', 'hum', 'visit_start', 'nectar_empty', 'visit_end', 'start_empty', 'start_end', 'hit_count'])
    
    for item in visit_info:
        morph_write.writerow(item)
    