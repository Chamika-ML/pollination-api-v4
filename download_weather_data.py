#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import requests
import time
from datetime import datetime,timezone,timedelta
import pytz #time zone data

import warnings
warnings.filterwarnings('ignore')
#commented all the #print functions because the api not working in aws deployment when we have #print functions
import multiprocessing

NO_OF_THREADS = 5


# In[2]:


def user_input_boundaries_to_latlong(lat_boundaries,long_boundaries):
    
    """This is the moest updated user input function. This funtion takes the user data in the form of lat long boundry arays
    and then returns the point grid varctors of given latitude and longitude boundaries"""
    
    box_lat_list, box_long_list = find_box(lat_boundaries,long_boundaries)
    
    # extract the data from user inputs    
    start_lat = max(box_lat_list)
    end_lat = min(box_lat_list)
    start_long = min(box_long_list)
    end_long = max(box_long_list)

    separation_meters = 70
    factor = 0.001 # for get points same as Qgis
    separation_degrees = separation_meters/111000  #One degree of latitude is approximately 111 kilometers
    num_of_points_lat = round(((abs(end_lat - start_lat)/factor) + 1))
    num_of_points_long = round(((abs(end_long - start_long)/factor) + 1))



    latitudes_arr = np.linspace(start_lat,end_lat,num_of_points_lat)
    longitudes_arr = np.linspace(start_long,end_long,num_of_points_long)

    # create grid points 
    point_grid = [(lat,long) for lat in latitudes_arr for long in longitudes_arr]
    latitudes = np.array([point_grid[i][0] for i in range(len(point_grid))])
    longitudes = np.array([point_grid[i][1] for i in range(len(point_grid))])
    
    # here longitudes_arr array containing the number of points in x direction (columns)
    # here latitudes_arr array containing the number of points in ydirection (raws)
    
    return latitudes,longitudes,len(longitudes_arr),len(latitudes_arr)

def temporal_processing_time_parallel(lat_boundaries,long_boundaries,speed_up=4,threads=NO_OF_THREADS,api_speed=60):
    """this function returns approximation time to download the weather data"""
    latitudes,longitudes,cols, raws = user_input_boundaries_to_latlong(lat_boundaries,long_boundaries)

    num_of_points = cols*raws
    download_points = round(num_of_points/speed_up)
    time_to_batch = round(download_points/threads)
    approx_time_mins = 2*round(time_to_batch/api_speed)  # 2  a is experimental value
    if approx_time_mins==0:
        approx_time_mins=1
        
    return approx_time_mins

def find_box(lat_boundaries,long_boundaries):
    
    """This function takes user boundaries and then finds and plots the square coodinates for convers the entier farm with 
    user boundaries. using the outputs we can find the max min lat long coordinates"""
    
    # to make the enclosed boundry 
    lat_boundaries[-1] = lat_boundaries[0]
    long_boundaries[-1] = long_boundaries[0]
    
    box_long_list = [min(long_boundaries),max(long_boundaries), max(long_boundaries), min(long_boundaries)]
    box_lat_list = [min(lat_boundaries),min(lat_boundaries),  max(lat_boundaries), max(lat_boundaries)]
    
    #if need to show plots then uncomment
    # for plot a complete square
    """box_long_list_plot = box_long_list.append(box_long_list[0])
    box_lat_list_plot = box_lat_list.append(box_lat_list[1])
    
    #plot the user boundaries and box boundaries
    plt.scatter(long_boundaries,lat_boundaries)
    plt.scatter(box_long_list,box_lat_list)

    plt.plot(long_boundaries,lat_boundaries, 'b')
    plt.plot(box_long_list,box_lat_list,'r')"""
    
    return box_lat_list,box_long_list

def download_weather_data_raw(latitudes,longitudes,cols,api,speed_up=4):
    
    # this function extract the weather data from api when provide the lat and long arrays (each raw of latitude)
    # the speed_up factor  determines that how may weather data values paeted by previous copied value, here it is pasted 3 values (4-1=3) by previous copied value.
    # here cols means number of points in a raw
    # create a data frame
    grid_point_Weather_data = pd.DataFrame(columns=["time","longitude", "latitude","tempreture", "humidity","wind_speed","weather_id", "weather_id_group", "weather_id_description", "sunrise", "sunset"])
    srt_time  = datetime.now()
    piangil_timezone = pytz.timezone('Australia/Sydney')

    for i in range(int(cols/speed_up)): # contralls the amount of the data

        srt_time_point  = datetime.now()
        #get the lat long coordinates
        lat = latitudes[speed_up*i] 
        long = longitudes[speed_up*i]

        #API url
        url = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric".format(lat,long,api)
        ##print(srt_time_point)
        ##print(url)

        piangil_time = datetime.now(piangil_timezone) #get time in Australia for data set
        
        #get data form API as json data (here wile loop is used to prevent to SSL erro failers)
        loop_though = True
        while loop_though:  
            try:
                res = requests.get(url)
                loop_though = False
            except:
                pass
        data = res.json()

        # create the data list that we want from the json data 
        data_vec = [piangil_time,long, lat, data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_1 = [piangil_time,longitudes[speed_up*i+1], latitudes[speed_up*i+1], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_2 = [piangil_time,longitudes[speed_up*i+2], latitudes[speed_up*i+2], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_3 = [piangil_time,longitudes[speed_up*i+3], latitudes[speed_up*i+3], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]


        #update the data frame
        grid_point_Weather_data.loc[speed_up*i] = data_vec
        grid_point_Weather_data.loc[speed_up*i+1] = data_vec_1
        grid_point_Weather_data.loc[speed_up*i+2] = data_vec_2
        grid_point_Weather_data.loc[speed_up*i+3] = data_vec_3

        # if the longitudes arr length (or raw length of the map points) can not divide by speed_up then remaining point in the columns should be filled previous values
        if(i%((int(cols/speed_up))-1)==0) and (cols%speed_up !=0) and (i!=0):
            num = cols%speed_up
            for j in range(num):
                data_vec_j = [piangil_time,longitudes[speed_up*i+3+(j+1)], latitudes[speed_up*i+3+(j+1)], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
                grid_point_Weather_data.loc[speed_up*i+3+(j+1)] = data_vec_j
                #print(f"this is done when step is equals to {i+1}")


        time.sleep(0.175)
        end_time_point  = datetime.now()
        #print(f"step {i+1} is completed! and taken {end_time_point-srt_time_point} time to complete")


    end_time = datetime.now()
    total_execution_time = end_time-srt_time
    #print(f"the programe take: {total_execution_time} to complete")
    
          
    return grid_point_Weather_data


def download_weather_data(latitudes,longitudes,cols,raws,api,key,return_dict):
    
    grid_point_Weather_data = pd.DataFrame(columns=["time","longitude", "latitude","tempreture", "humidity","wind_speed","weather_id", "weather_id_group", "weather_id_description", "sunrise", "sunset"])
    
    for i in range(raws):
        
        # selecting each raw of latitude and longitude arrays
        lat_arr = latitudes[i*cols:(i+1)*cols]
        long_arr = longitudes[i*cols:(i+1)*cols] 
        
        # get weather data for each raw of latitudes and longitudes
        first_batch_data = download_weather_data_raw(lat_arr,long_arr,cols,api)
        
        # combine the pandas dataframe with previoues one
        grid_point_Weather_data = pd.concat([grid_point_Weather_data,first_batch_data], axis=0, ignore_index=True)
        #print(f"complete the {i+1} raw data download")
        #print("==================")
        #print("==================")
    
    # set the Id column and charge the raw order
    grid_point_Weather_data["id"] = [j+1 for j in range(cols*raws)]
    grid_point_Weather_data = grid_point_Weather_data[["id","time","longitude", "latitude","tempreture", "humidity","wind_speed","weather_id", "weather_id_group", "weather_id_description", "sunrise", "sunset"]]
    
    return_dict[key] = grid_point_Weather_data


def unix_to_aus(time):
    
    """this function convert UNIX date time to Austrelia date time and output will be string. This function is called
    inside the download_weather_data_raw function """
    
    time_int = int(time) #get integer value
    
    time_zone = timezone(timedelta(seconds=36000)) # time zone of Austrelia 
    
    aus_time = datetime.fromtimestamp(time_int, tz = time_zone).strftime('%Y-%m-%d %H:%M:%S')
    #aus_time = datetime.fromtimestamp(time_int, tz = time_zone)
    
    return aus_time

def lat_long_batches(latitudes,longitudes,cols,raws,threads=NO_OF_THREADS):
    """This function takes lat long grid points, num of raws and cols and then returns lat long points batches to call parallel api callings"""
    long_batches = []
    lat_batches = []
    batch_raw = int(raws/threads)+1
    batch_range = batch_raw*cols

    for i in range(threads):
        long_batch = longitudes[i*batch_range:(i+1)*batch_range]
        lat_batch = latitudes[i*batch_range:(i+1)*batch_range]

        long_batches.append(long_batch)
        lat_batches.append(lat_batch)

    return lat_batches,long_batches


def create_weather_dataset(lat_boundaries,long_boundaries,api_keys):
    """This is the final function that we need to call to download the weather dataset"""   
    latitudes,longitudes,cols, raws = user_input_boundaries_to_latlong(lat_boundaries,long_boundaries)
    lat_batches,long_batches =  lat_long_batches(latitudes,longitudes,cols,raws)
            
    manager = multiprocessing.Manager()
    return_dict = manager.dict() # this dict contains the results of all threads
    jobs = [] # this list contains the all threads
    
    for i in range(NO_OF_THREADS):
        p = multiprocessing.Process(target=download_weather_data, args=(lat_batches[i],long_batches[i],cols,int(len(lat_batches[i])/cols),api_keys[i],i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    result_list = []
    for i in range(len(return_dict)):
        result_list.append(return_dict[i])
        
    dataset = pd.concat(result_list)
    #preprocess the dataset
    dataset.drop(["id"], axis=1, inplace=True)
    id_col = np.arange(1,len(dataset)+1,1)
    dataset["id"] = id_col
    dataset.set_index([np.arange(0,len(dataset),1)], inplace=True)
        
    #dataset.to_csv("./results/csv/weather_dataset.csv", index=False)
    return dataset,latitudes,longitudes,cols,raws

