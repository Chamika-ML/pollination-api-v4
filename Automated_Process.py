#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from geopy.distance import geodesic

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import folium
from folium.raster_layers import ImageOverlay
from folium import plugins

import requests
import time
from datetime import datetime,timezone,timedelta
import pytz #time zone data

import exifread
import os
import warnings
warnings.filterwarnings('ignore')
#commented all the #print functions because the api not working in aws deployment when we have #print functions

#!pip install --upgrade requests # to overcome the ssl error in downloading

#!pip install mysql-connector-python
import mysql.connector
#!pip install sqlalchemy
from sqlalchemy import create_engine, inspect

from download_weather_data import create_weather_dataset,temporal_processing_time_parallel
from datetime import datetime

#PDF_PI_FILE_PATH = "./results/csv/grid_pdf_pi.csv"
#HIVE_DETAILS_FILE_PATH = "./data/csv/hive_detailss.csv"
#WEATHER_DESCRIPTION_FILE_PATH = "./data/csv/weather_description_map.csv"
#FINAL_WEATHER_DATA_FILE_PATH = "./results/csv/final_weather_data.csv"
SPATIAL_MAP_SAVE_PATH = "./results/maps/spatial_map.html"
FINAL_MAP_SAVE_PATH = "./results/maps/final_map.html"
MS_TO_KMH = 3.6

#HIVE_DETAILS_QUERY = "SELECT * FROM hive_details"
#WEATHER_DESCRIPTION_QUERY = "SELECT * FROM weather_description_map"
#PDF_PI_FILE_QUERY = "SELECT * FROM grid_pdf_pi"
#FINAL_WEATHER_DATA_FILE_QUERY = "SELECT * FROM final_weather_data"

WEATHER_DESCRIPTION_TABLE = "weather_description_map"
HIVE_DETAILS_TABLE_PREFIX = "hive_details"
PDF_PI_TABLE_PREFIX = "grid_pdf_pi"
FINAL_WEATHER_TABLE_PREFIX = "final_weather_data"
MAPS_TABLE_PREFIX = "maps"

MYSQL_CREDENTIALS = {"host":"127.0.0.1", "user":"dilshan", "password":"1234", "database":"broodbox", "port":3306}
#MYSQL_CREDENTIALS = {"host":"127.0.0.1", "user":"root", "password":"", "database":"broodbox", "port":3306}

# ## Database Functions 

# In[2]:


def read_data_from_mesql(TABLE_NAME,credentials=MYSQL_CREDENTIALS, last_raw=False):
    
    """this function uses to read mysql table and returns it as a dataframe. if last_raw=True then this will returns last raw of the table"""
    # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=credentials["host"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"]
    )
    
    if last_raw:
        query = f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC LIMIT 1"
    else:
        query = f"SELECT * FROM {TABLE_NAME}"
    # Create a cursor object to interact with the database
    cursor = connection.cursor()
    
    # Execute the query
    cursor.execute(query)

    # Fetch all rows from the result set
    rows = cursor.fetchall()
    
    # Convert the data to a Pandas DataFrame
    column_names = [i[0] for i in cursor.description]
    df = pd.DataFrame(rows, columns=column_names)
    
    # Close the cursor and connection
    cursor.close()
    connection.close()
    
    return df


def create_mysql_table(dataset, table_name, credentials=MYSQL_CREDENTIALS):
    
    "this function creates a table in mysql database using pandas dataframe"
    
    engine = create_engine(f'mysql+mysqlconnector://{credentials["user"]}:{credentials["password"]}@{credentials["host"]}:{credentials["port"]}/{credentials["database"]}', connect_args={"connect_timeout": 28800})

    dataset.to_sql(table_name, con=engine, if_exists='replace', index=False)
    
    engine.dispose()
    
   
def table_exist_mysql_database(table_name, credentials=MYSQL_CREDENTIALS):
    
    "this function returns true if the given tables exists in the mysql database , other wise it rerurns false"
    # Create a MySQL connection
    engine = create_engine(f'mysql+mysqlconnector://{credentials["user"]}:{credentials["password"]}@{credentials["host"]}:{credentials["port"]}/{credentials["database"]}')

    # Check if the dataset exists
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Print the result
    if table_name in tables:
        exist = True
    else:
        exist = False

    # Close the connection (optional)
    engine.dispose()

    return exist


def insert_values_totable(table_name,spatial_map,final_map,credentials=MYSQL_CREDENTIALS):

    """This function inserts data to maps table. if maps tables not exist then first it will creates a maps table and then insert the data"""
     # Connect to the MySQL server
    connection = mysql.connector.connect(
        host=credentials["host"],
        user=credentials["user"],
        password=credentials["password"],
        database=credentials["database"]
    )

    cursor = connection.cursor()
    
    if not table_exist_mysql_database(table_name, credentials=MYSQL_CREDENTIALS):
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE,
            time TIME,
            spatial_map LONGTEXT,
            final_map LONGTEXT 
        )
        """
        cursor.execute(create_table_sql)
        connection.commit()

    current_date = datetime.now().date()
    current_time = datetime.now().time()

    insert_sql = f"""
    INSERT INTO {table_name} (date, time, spatial_map, final_map)
    VALUES (%s, %s, %s, %s)
    """

    cursor.execute(insert_sql, (current_date, current_time, spatial_map, final_map))

    connection.commit()
    cursor.close()
    connection.close()

    
# ## Spatial Functions 

# In[3]:


def diffrence_between_tow_points(lat1, lon1, lat2, lon2):
    
    """This funtion finds the distance between two locations in Km when the longitudes and latitudes of the two points are given"""
    
    R = 6371 # radius of the eatch in kilo meters 
    lon1_rad = math.radians(lon1) # convert degrees to radians
    lon2_rad = math.radians(lon2)    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)   
    
    del_lon = lon2_rad - lon1_rad
    del_lat = lat2_rad - lat1_rad
    
    a = (math.sin(del_lat/2))**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*((math.sin(del_lon/2))**2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    
    return d


def distence_probability(dist):
    
    """Takes one distencs and convert it to probability"""
    
    # Distence probability function
    d1 = (np.exp(dist/100))/15.6
    d2 = 1.63/d1 
    prob = np.where(dist<275,d1,np.where(dist<=325,1,np.where(dist<=500,d2,0)))
         
    return prob


def distance_matrix(latitudes,longitudes,hive_details_dataset):
    
    """this function finds distence between each grid point and hive location""" 
    
    grid_point_latitudes = latitudes
    grid_point_longitudes = longitudes

    hive_point_latitudes = np.array(hive_details_dataset['latitude'])
    hive_point_longitudes = np.array(hive_details_dataset['longitude']) 
    
    distance_arr = []

    for i in range(len(grid_point_latitudes)):

        point1 = (grid_point_latitudes[i],grid_point_longitudes[i])

        distance_vec = []

        for j in range(len(hive_point_latitudes)):

            point2 = (hive_point_latitudes[j],hive_point_longitudes[j])

            distance = geodesic(point1, point2).meters
            distance_vec.append(distance)

        distance_arr.append(distance_vec)
        
    return np.array(distance_arr)


def probability_matrix(distance_matrix,hive_details_dataset):
    
    """This function convert the distences to probabilities"""
    
    total_frames = np.array(hive_details_dataset['total_active_frames'])
    
    prob_dist = [] # this is 2d vector containing all the probabilities of points form hives (3960*260)

    for i in range(distance_matrix.shape[0]):

        prob_dist_vec = [] # probabilities containing each grid point in a row (len 260)

        for j in range(distance_matrix.shape[1]):

            point_dist = distance_matrix[i][j] # distance from hive
            prob = distence_probability(point_dist)

            # append the (probability*total frames) crosponding distance range
            prob_dist_vec.append(prob*total_frames[j])


        prob_dist.append(prob_dist_vec) 

    prob_dist_arr = np.array(prob_dist)
    
    return prob_dist_arr


def convert_one_probability(prob_dist_arr):
    
    """This function takes matrics of probabilities and gives the sum of each raw them"""
    # get the sum of each rows in the probability metrix
    sum_prob_vec = [] # get the sum of all raws 

    for i in range(prob_dist_arr.shape[0]):

        distance_row = prob_dist_arr[i]

        sum_distance_row = np.sum(distance_row)

        sum_prob_vec.append(sum_distance_row)

    sum_prob_arr = np.array(sum_prob_vec) 

    norm_sum_prob_arr = (sum_prob_arr - np.min(sum_prob_arr))/(np.max(sum_prob_arr)-np.min(sum_prob_arr)) #  normalized sum_prob_arr using min max formula

    return norm_sum_prob_arr


def spatial_probability_dataset(lat,long,HIVE_DETAILS_TABLE,PDF_PI_TABLE):
    
    """This is the finla function. this function call all the above funcions to make the sptial probabilities"""
    
    hive_details_dataset = read_data_from_mesql(HIVE_DETAILS_TABLE)
    
    distances = distance_matrix(lat,long,hive_details_dataset)
    porbabilities = probability_matrix(distances,hive_details_dataset)  
    norm_sum_prob_arr = convert_one_probability(porbabilities)
    
    data = {'id':np.arange(1,len(norm_sum_prob_arr)+1,1), 'longitude':long,'latitude':lat, 'spatial_prob':norm_sum_prob_arr}
    dataset = pd.DataFrame(data)
    create_mysql_table(dataset, PDF_PI_TABLE)
    


# ## Weather Functions

# #### Weather Probability functions

# In[4]:


def tempreture_probability(tempreture):

    t1= 0.141*np.exp(tempreture/10) - 0.1
    t2 = (3.4/t1)-0.42
    prob = np.where(tempreture<0,0,np.where(tempreture<=20,t1,np.where(tempreture<=30,1,np.where(tempreture<=40,t2,0))))
    
    return prob

def humidity_probability(humidity):
    
    h1 = 0.0322*np.exp(humidity/10)
    h2 = 2.7/h1 
    prob = np.where(humidity<35,h1,np.where(humidity<=45,1,h2))
    
    return prob

def wind_probability(speed):
                         
    w1 = 3*np.exp(-speed/10) - 0.15
    prob = np.where(speed<=10,1,np.where(speed>=30,0,w1))
    return prob

def hour_probability(data_set):
    
    """This function returns 1 if the sun in sky, otherwise gives 0"""
    
    hour_prob = []
    
    for i in range(data_set.shape[0]):
        
        #get times as strings
        sunrise_str = str(data_set["sunrise"][i]).split()[1]
        sunset_str = str(data_set["sunset"][i]).split()[1]
        curr_time_str = str(data_set["time"][i])[:8]
        
        #get time strings as time objects
        sunrise = datetime.strptime(sunrise_str, '%H:%M:%S').time()
        sunset = datetime.strptime(sunset_str, '%H:%M:%S').time()
        curr_time = datetime.strptime(curr_time_str, '%H:%M:%S').time()
        
        #checks the current time and sunset and sunrise
        if(sunrise<=curr_time<=sunset):
            hour_prob.append(1.0)
        else:
            hour_prob.append(0.0)
            
    return hour_prob

def should_spatial_probability_call(file_path):
    """this function returns true if the hive_details file not exist or it was updated withing 5 mins otherwise it returns false"""
    update = False    
    # Check if the file not exists updated before 300 seconnds
    if not os.path.exists(file_path):
        update = True
        return update
    
    current_time = datetime.now().timestamp()
    # Get file metadata covert it to seconds using timestamp()
    modification_time = datetime.fromtimestamp(os.path.getmtime(file_path)).timestamp()
    
    # Check if the file updated before 300 seconnds
    if (int(current_time -modification_time) <= 300):
        update = True
   
    return update
                         
def final_probability(data_set,lat,long,PDF_PI_TABLE,HIVE_DETAILS_TABLE):
    
    table_exist = table_exist_mysql_database(PDF_PI_TABLE)
    #load spatial porbability data set 
    if table_exist==False:
        spatial_probability_dataset(lat,long,HIVE_DETAILS_TABLE,PDF_PI_TABLE)
        spatial_prob_data = read_data_from_mesql(PDF_PI_TABLE)
    else:
        spatial_prob_data = read_data_from_mesql(PDF_PI_TABLE)
      
    #load weather description porbability data set
    weather_desc_data = read_data_from_mesql(WEATHER_DESCRIPTION_TABLE)
    # genarate weather probability using mean ratings
    weather_desc_data["probability"] = (weather_desc_data["mean_ratings"]-1)/(10-1)

    data_set["weather_condition_prob"] = list(weather_desc_data[weather_desc_data["weather_id"]==list(data_set["weather_id"])[0]]["probability"])[0]

    
    #hour probability
    hour_prob_arr = hour_probability(data_set)
    data_set["hour_prob"] = hour_prob_arr
    
    data_set["tempreture_prob"] = data_set["tempreture"].apply(tempreture_probability)
    data_set["humidity_prob"] = data_set["humidity"].apply(humidity_probability)
    data_set["wind_prob"] = data_set["wind_speed"].apply(wind_probability)
                                    
                                    
    prob = np.array(data_set["tempreture_prob"]*data_set["humidity_prob"]*data_set["wind_prob"]* data_set["weather_condition_prob"]*data_set["hour_prob"])
    data_set["weather_prob"] = prob
    
    final_data_set = pd.merge(data_set,spatial_prob_data, on='id')
    final_data_set["final_prob"] = final_data_set["weather_prob"]*final_data_set["spatial_prob"]
    final_data_set.drop(columns=["longitude_y","latitude_y"], axis=1, inplace = True)
    final_data_set.rename(columns = {'longitude_x':'longitude', 'latitude_x':'latitude'}, inplace = True)
    
    
    return final_data_set
    


# #### Weather data donwload functions 

# In[5]:


def download_weather_data_raw(latitudes,longitudes,cols,speed_up=4):
    
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
        url = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid=8ee842d65cf08ec205365865e3d53348&units=metric".format(lat,long)
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
        data_vec = [piangil_time,long, lat, data["main"]["temp"], data["main"]["humidity"], MS_TO_KMH*data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_1 = [piangil_time,longitudes[speed_up*i+1], latitudes[speed_up*i+1], data["main"]["temp"], data["main"]["humidity"], MS_TO_KMH*data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_2 = [piangil_time,longitudes[speed_up*i+2], latitudes[speed_up*i+2], data["main"]["temp"], data["main"]["humidity"], MS_TO_KMH*data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_3 = [piangil_time,longitudes[speed_up*i+3], latitudes[speed_up*i+3], data["main"]["temp"], data["main"]["humidity"], MS_TO_KMH*data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]


        #update the data frame
        grid_point_Weather_data.loc[speed_up*i] = data_vec
        grid_point_Weather_data.loc[speed_up*i+1] = data_vec_1
        grid_point_Weather_data.loc[speed_up*i+2] = data_vec_2
        grid_point_Weather_data.loc[speed_up*i+3] = data_vec_3

        # if the longitudes arr length (or raw length of the map points) can not divide by speed_up then remaining point in the columns should be filled previous values
        if(i%((int(cols/speed_up))-1)==0) and (cols%speed_up !=0) and (i!=0):
            num = cols%speed_up
            for j in range(num):
                data_vec_j = [piangil_time,longitudes[speed_up*i+3+(j+1)], latitudes[speed_up*i+3+(j+1)], data["main"]["temp"], data["main"]["humidity"], MS_TO_KMH*data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
                grid_point_Weather_data.loc[speed_up*i+3+(j+1)] = data_vec_j
                #print(f"this is done when step is equals to {i+1}")


        time.sleep(0.175)
        end_time_point  = datetime.now()
        #print(f"step {i+1} is completed! and taken {end_time_point-srt_time_point} time to complete")


    end_time = datetime.now()
    total_execution_time = end_time-srt_time
    #print(f"the programe take: {total_execution_time} to complete")
    
          
    return grid_point_Weather_data


def download_weather_data_old(latitudes,longitudes,cols,raws):
    
    grid_point_Weather_data = pd.DataFrame(columns=["time","longitude", "latitude","tempreture", "humidity","wind_speed","weather_id", "weather_id_group", "weather_id_description", "sunrise", "sunset"])
    
    for i in range(raws):
        
        # selecting each raw of latitude and longitude arrays
        lat_arr = latitudes[i*cols:(i+1)*cols]
        long_arr = longitudes[i*cols:(i+1)*cols] 
        
        # get weather data for each raw of latitudes and longitudes
        first_batch_data = download_weather_data_raw(lat_arr,long_arr,cols)
        
        # combine the pandas dataframe with previoues one
        grid_point_Weather_data = pd.concat([grid_point_Weather_data,first_batch_data], axis=0, ignore_index=True)
        #print(f"complete the {i+1} raw data download")
        #print("==================")
        #print("==================")
    
    # set the Id column and charge the raw order
    grid_point_Weather_data["id"] = [j+1 for j in range(cols*raws)]
    grid_point_Weather_data = grid_point_Weather_data[["id","time","longitude", "latitude","tempreture", "humidity","wind_speed","weather_id", "weather_id_group", "weather_id_description", "sunrise", "sunset"]]
    
    return grid_point_Weather_data


# #### Weather data preprocess functions

# In[6]:


def unix_to_aus(time):
    
    """this function convert UNIX date time to Austrelia date time and output will be string. This function is called
    inside the download_weather_data_raw function """
    
    time_int = int(time) #get integer value
    
    time_zone = timezone(timedelta(seconds=36000)) # time zone of Austrelia 
    
    aus_time = datetime.fromtimestamp(time_int, tz = time_zone).strftime('%Y-%m-%d %H:%M:%S')
    #aus_time = datetime.fromtimestamp(time_int, tz = time_zone)
    
    return aus_time


def add_date_time(dataset):
    
    """This function add a date column and time column for a given pandas dataframe using Sunrice column data
     and Time column data."""
    
    # create a date column as first column
    date_column = dataset["sunrise"].apply(lambda x: ((str(x)).split())[0])
    dataset.insert(1, "date",date_column)

    # update the Time column
    dataset["time"] = dataset["time"].apply(lambda x: (str(x)).split()[1][:11])
    
    return dataset


# ## Image taken locations plot functions 

# In[7]:


def read_image_metadata_exif(image_path):
    """his function returns the meta data ofimages"""
    try:
        with open(image_path, 'rb') as img_file:
            # Get Exif tags
            tags = exifread.process_file(img_file)
            return tags
    except Exception as e:
        ##print(f"Error: {e}")
        return None

def convert_lat_long(list_lat,list_long):
    """This function convert lat long meta deta to actual lat long coordinates"""
    #decimal degrees = degrees + minutes / 60 + seconds / 3600
    if list_lat == "" or list_long == "":
        lat  = np.NaN
        long = np.NaN
    
    else:
        lat = -(list_lat[0] + list_lat[1]/60 + list_lat[2]/3600)
        long = list_long[0] + list_long[1]/60 + list_long[2]/3600 
    
    return lat,long

def image_taken_location_dataset(img_folder_path):
    """This function loop througth each image and extract meta data and find lat long and frame counts and make pandas table"""
    
    image_names = os.listdir(img_folder_path)
    image_psths = [f"./images/images/{img}" for img in image_names]

    frame_count_arr = []
    lat_arr = []
    long_arr = []

    frame_key = "Image ImageDescription" 
    lat_key = "GPS GPSLatitude"
    long_key = "GPS GPSLongitude"

    for img_pth in image_psths:

        result = read_image_metadata_exif(img_pth)

        if frame_key in result:
            frame_count = str(result["Image ImageDescription"])
        else:
            frame_count = np.NaN

        if lat_key in result:
            lat_list = eval(str(result["GPS GPSLatitude"]))
        else:
            lat_list = ""

        if long_key in result:
            long_list = eval(str(result["GPS GPSLongitude"]))
        else:
            long_list = ""

        lat,long = convert_lat_long(lat_list,long_list)
        frame_count_arr.append(frame_count)
        lat_arr.append(lat)
        long_arr.append(long)

    data = {"lat":lat_arr, "long":long_arr, "frame_count":frame_count_arr}
    dataset = pd.DataFrame(data)
    updated_dataset = dataset.dropna()
    return updated_dataset


def location_grid_frame_count(img_folder_path):
    """This function create lat long coordinate pairs for each location and preprocess the frame count of the pandas data frame and returns"""
    
    updated_dataset = image_taken_location_dataset(img_folder_path)
    location = list(zip(updated_dataset["lat"],updated_dataset["long"]))
    frame_count = updated_dataset["frame_count"].values
    frame_count[608] = '4/4,10/10' #this point's original value contains error for the map
    
    return location,frame_count


# ## User Input Functions 

# In[8]:


def user_input_to_latlong_old():
    
    """This funtion takes the user data  in form of latitudes and longitudes like this: max latitude,min latitude, min longitude, max longitude
    and return the point grid varctors of given latitude and longitude boundaries"""

    user_input = input("Enter the Lat Long codinates separated by a comma:")

    # get and evaluate the user inputs
    try:
        splited_input = user_input.split(",")

        # user can only enter four numbers
        if(len(splited_input)==4):
            start_latitude = float(splited_input[0])
            end_latitude = float(splited_input[1])

            start_longitude = float(splited_input[2])
            end_longitude = float(splited_input[3])

            ##print(f"Your start and end latitudes are:{[start_latitude,end_latitude]} and start and end longitudes are:{[start_longitude,end_longitude]}")
        else:
            pass
            ##print("Exceed or less number of inputes. Check the inputs again.")


    except (ValueError,IndexError):
        pass
        ##print("Error! Invalid input. Please enter valied input")


    # extract the data from user inputs    
    start_lat = start_latitude
    end_lat = end_latitude
    start_long = start_longitude
    end_long = end_longitude


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


def api_to_latlong(lat_boundaries,long_boundaries):
    
    """Thsi function takes  lat longs boundaries form the api and returns the point grid varctors of given latitude and longitude boundaries""" 
    start_lat,end_lat,start_long,end_long = max(lat_boundaries), min(lat_boundaries), min(long_boundaries), max(long_boundaries)
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
    


# ## User Output Functions

# #### Masking functions 

# In[9]:


def find_box(lat_boundaries,long_boundaries):
    
    """This function takes user boundaries and then finds and plots the square coodinates for convers the entier farm with 
    user boundaries. using the outputs we can find the max min lat long coordinates"""
    
    # to make the enclosed boundry 
    lat_boundaries[-1] = lat_boundaries[0]
    long_boundaries[-1] = long_boundaries[0]
    
    box_long_list = [min(long_boundaries),max(long_boundaries), max(long_boundaries), min(long_boundaries)]
    box_lat_list = [min(lat_boundaries),min(lat_boundaries),  max(lat_boundaries), max(lat_boundaries)]
    
    #if need to show plots then uncomment
    """# for plot a complete square
    box_long_list_plot = box_long_list.append(box_long_list[0])
    box_lat_list_plot = box_lat_list.append(box_lat_list[1])
    
    #plot the user boundaries and box boundaries
    plt.scatter(long_boundaries,lat_boundaries)
    plt.scatter(box_long_list,box_lat_list)

    plt.plot(long_boundaries,lat_boundaries, 'b')
    plt.plot(box_long_list,box_lat_list,'r')"""
    
    return box_lat_list,box_long_list


def make_mask_fromm_boundaries(lats,longs,lat_boundaries,long_boundaries):
    """This function takes lats longs form the data frame and user farm boundaries then creates a mask array using user enterd 
    boundaries """

    mask = []
    boundaries = list(zip(long_boundaries,lat_boundaries)) # get each points as list
    polygon = Polygon(boundaries) # create a polygon usign boundaries

    for i in range(len(lats)):

        long = longs[i]
        lat = lats[i] 
        point = Point(long,lat)

        # if the point inside the polygon
        if(polygon.contains(point)):
            mask.append(1)
        else:
            mask.append(0)
            
    return np.array(mask)


def make_mask_dataset(lat_boundaries,long_boundaries,data_set):
    
    """This function removes the data points that are not withing user boundaries. this is the function that we need to call to create final data set
    when we are using the uer defined boundaries. above two masked functions used here"""
    
    box_lat_list, box_long_list = find_box(lat_boundaries,long_boundaries)
    
    lats = data_set["latitude"].values
    longs = data_set["longitude"].values
    # create a mask
    mask = make_mask_fromm_boundaries(lats,longs,lat_boundaries,long_boundaries)

    #selected the points withn the user boundry and update the table 
    data_set["mask"] = mask
 
    masked_data_set = data_set.iloc[data_set[data_set["mask"]==1].index].drop(["mask"], axis=1)
    #if need to show plots then uncomment
    """#plot the selected lat long points using mask
    plt.scatter(masked_data_set["longitude"],masked_data_set["latitude"],2)
    plt.title("The user defined boundaries and selected data points to show result")"""
   
    return masked_data_set


# In[10]:


def spatial_processing_time(lat_boundaries,long_boundaries,HIVE_DETAILS_TABLE):
    """this function returns approximation time to calculate spatial probabilities"""
    latitudes,longitudes,cols, raws = user_input_boundaries_to_latlong(lat_boundaries,long_boundaries)
    hive_details_dataset = read_data_from_mesql(HIVE_DETAILS_TABLE)
    no_of_locations = len(hive_details_dataset)
    num_of_points = no_of_locations*cols*raws

    approx_time_mins = int((num_of_points/4000)/60) # 4000 a is experimental value
    if approx_time_mins==0:
        approx_time_mins=1
        
    return approx_time_mins

def temporal_processing_time(lat_boundaries,long_boundaries,speed_up=4,api_speed=60):
    """this function returns approximation time to download the weather data"""
    latitudes,longitudes,cols, raws = user_input_boundaries_to_latlong(lat_boundaries,long_boundaries)

    num_of_points = cols*raws
    download_points = round(num_of_points/speed_up)
    approx_time_mins = round(download_points/api_speed)  
    if approx_time_mins==0:
        approx_time_mins=1
        
    return approx_time_mins

def spatial_heatmap(lat,long,lat_boundaries, long_boundaries,HIVE_DETAILS_TABLE,PDF_PI_TABLE,image_path=" "):
    """This function returns the spatial heatmap"""  
    #load spatial porbability data set
    table_exist = table_exist_mysql_database(PDF_PI_TABLE)
    #load spatial porbability data set 
    if table_exist==False:
        spatial_probability_dataset(lat,long,HIVE_DETAILS_TABLE,PDF_PI_TABLE)
        dataset = read_data_from_mesql(PDF_PI_TABLE)
    else:
        dataset = read_data_from_mesql(PDF_PI_TABLE)

    #applying masking
    dataset = make_mask_dataset(lat_boundaries,long_boundaries,dataset)
    if dataset.empty == True:
        m = np.NaN
        return m
         
    longitudes = dataset["longitude"]
    latitudes = dataset["latitude"]
    probability = dataset["spatial_prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   max_zoom=15,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)
    
    """# #print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        pass
        ##print("Please provide a valid image path")

    return m

def temporal_heatmap(lat_boundaries, long_boundaries, dataset, image_path=" "):
    """This function returns the temporal heatmap"""
    
    #applying masking
    dataset = make_mask_dataset(lat_boundaries,long_boundaries,dataset)
    if dataset.empty == True:
        m = np.NaN
        return m
    
    longitudes = dataset["longitude"]
    latitudes = dataset["latitude"]
    probability = dataset["weather_prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   max_zoom=15,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)
    
    """# #print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        pass
        ##print("Please provide a valid image path")

    return m


def final_heatmap_with_image_locations(lat_boundaries, long_boundaries, dataset, image_path=" "):
    """This function returns the final heatmap with image taken locations with markers"""
    location, frame_count = location_grid_frame_count("./images/images/")
    
    #applying masking
    dataset = make_mask_dataset(lat_boundaries,long_boundaries,dataset)
    if dataset.empty == True:
        m = np.NaN
        return m
    
    longitudes = dataset["longitude"]
    latitudes = dataset["latitude"]
    probability = dataset["final_prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   max_zoom=15,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)

    # Add markers for each coordinate in the list
    for i, coord in enumerate(location):
        folium.Marker(location=coord, popup=f'{frame_count[i]}').add_to(m)
        
    """# #print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        pass
        ##print("Please provide a valid image path")

    return m

def final_heatmap(lat_boundaries, long_boundaries, dataset, image_path=" "):
    
    """This function returns the final heatmap"""
    
    #applying masking
    dataset = make_mask_dataset(lat_boundaries,long_boundaries,dataset)
    if dataset.empty == True:
        m = np.NaN
        return m
    
    longitudes = dataset["longitude"]
    latitudes = dataset["latitude"]
    probability = dataset["final_prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   max_zoom=15,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)
     
    """# #print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        pass
        ##print("Please provide a valid image path")

    return m


# ## Final calling Functions 

# In[11]:


def final_maps_api(lat_boundaries,long_boundaries,PDF_PI_TABLE,FINAL_WEATHER_TABLE,HIVE_DETAILS_TABLE):    
    """This is the final fuction that we need to call when we are using api when we use user defined boundaries"""
    
    box_lat_list, box_long_list = find_box(lat_boundaries,long_boundaries)
    
    # extract the data from user inputs    
    start_lat = max(box_lat_list)
    end_lat = min(box_lat_list)
    start_long = min(box_long_list)
    end_long = max(box_long_list)

    lat, long, cols, raws = api_to_latlong(lat_boundaries,long_boundaries)
    dataset = download_weather_data_old(lat,long,cols, raws)
    dataset = add_date_time(dataset)
    dataset = final_probability(dataset,lat,long,PDF_PI_TABLE,HIVE_DETAILS_TABLE)
       
    spatial_map = spatial_heatmap(lat,long,lat_boundaries, long_boundaries,HIVE_DETAILS_TABLE,PDF_PI_TABLE)  
    final_map = final_heatmap(lat_boundaries, long_boundaries,dataset)
    create_mysql_table(dataset, FINAL_WEATHER_TABLE)
    
    spatial_map.save(SPATIAL_MAP_SAVE_PATH)
    final_map.save(FINAL_MAP_SAVE_PATH)

    with open(SPATIAL_MAP_SAVE_PATH, 'r', encoding='utf-8') as file_sp:
        spatial_html_content = file_sp.read()
        
    with open(FINAL_MAP_SAVE_PATH, 'r', encoding='utf-8') as file_fn:
        finalmap_html_content = file_fn.read()

    return spatial_html_content,finalmap_html_content


def final_maps_api_parallel(lat_boundaries,long_boundaries,api_keys,bid,fid):

    """This is the final fuction that we need to call when we are using api when we use user defined boundaries and parallel downloading"""
    HIVE_DETAILS_TABLE = f"{HIVE_DETAILS_TABLE_PREFIX}_{bid}_{fid}"
    PDF_PI_TABLE = f"{PDF_PI_TABLE_PREFIX}_{bid}_{fid}"
    FINAL_WEATHER_TABLE = f"{FINAL_WEATHER_TABLE_PREFIX}_{bid}_{fid}"
    MAPS_TABLE = f"{MAPS_TABLE_PREFIX}_{bid}_{fid}"
    
    dataset,lat,long,cols,raws = create_weather_dataset(lat_boundaries,long_boundaries,api_keys)
    dataset = add_date_time(dataset)
    dataset = final_probability(dataset,lat,long,PDF_PI_TABLE,HIVE_DETAILS_TABLE)
       
    spatial_map = spatial_heatmap(lat,long,lat_boundaries, long_boundaries,HIVE_DETAILS_TABLE,PDF_PI_TABLE)  
    final_map = final_heatmap(lat_boundaries, long_boundaries,dataset)
    create_mysql_table(dataset, FINAL_WEATHER_TABLE)

    
    spatial_map.save(SPATIAL_MAP_SAVE_PATH)
    final_map.save(FINAL_MAP_SAVE_PATH)

    with open(SPATIAL_MAP_SAVE_PATH, 'r', encoding='utf-8') as file_sp:
        spatial_html_content_data = file_sp.read()
        
    with open(FINAL_MAP_SAVE_PATH, 'r', encoding='utf-8') as file_fn:
        finalmap_html_content_data = file_fn.read()
    
    # insert map values to the maps table
    insert_values_totable(MAPS_TABLE,spatial_html_content_data,finalmap_html_content_data)
    # remove saved maps on the server
    os.remove(SPATIAL_MAP_SAVE_PATH)
    os.remove(FINAL_MAP_SAVE_PATH)

    maps_data = read_data_from_mesql(MAPS_TABLE, last_raw=True)

    spatial_html_content = maps_data["spatial_map"].values[0]
    finalmap_html_content = maps_data["final_map"].values[0]
    map_id =  maps_data["id"].values[0]
    
    return spatial_html_content,finalmap_html_content, map_id


def final_maps(lat_boundaries,long_boundaries,api_keys,PDF_PI_TABLE,FINAL_WEATHER_TABLE,HIVE_DETAILS_TABLE):
    
    """This is the final fuction that we need to call when we are using this notebook"""

    time= temporal_processing_time(lat_boundaries,long_boundaries) + spatial_processing_time(lat_boundaries,long_boundaries,HIVE_DETAILS_TABLE)
    print(f"This will take {time} mins to complete")
    dataset,lat, long,cols,raws = create_weather_dataset(lat_boundaries,long_boundaries,api_keys)
    dataset = add_date_time(dataset)
    dataset = final_probability(dataset,lat,long,PDF_PI_TABLE,HIVE_DETAILS_TABLE)
    
    spatial_map = spatial_heatmap(lat,long,lat_boundaries, long_boundaries,HIVE_DETAILS_TABLE,PDF_PI_TABLE)  
    final_map = final_heatmap(lat_boundaries, long_boundaries,dataset)
    create_mysql_table(dataset, FINAL_WEATHER_TABLE)

    spatial_map.save(SPATIAL_MAP_SAVE_PATH)
    final_map.save(FINAL_MAP_SAVE_PATH)
    
    return spatial_map,final_map,dataset


"""# ### UI part 

# In[12]:


#list_a = [-35.083200762,-35.142200762,143.251973043,143.316973043]

api_keys = ["8ee842d65cf08ec205365865e3d53348", "c29f27459329c5bbffb6e633e0fc4502", "51fb82fd74e9c378a1983d2551733418", "94979cc6f8c54c197d859f25576fb942", "21492bb5c90d0b7156cb1c5c543cb3c2"]

lat_boundaries = [-35.08491940916005,-35.11377165513988, -35.11377165513988, -35.128093335964095, -35.12816353408006, -35.14255287012573,-35.14248268441524, -35.12876021562264, -35.11377165513988, -35.11363123404099, -35.098464331083484, -35.08475718058806, -35.08478375410176, -35.08486825609331]

long_boundaries = [143.2558918258238, 143.2560585113483, 143.28189354857975, 143.28193646392398, 143.29060536346012, 143.29051953277164, 143.3088443847614, 143.3087585540729, 143.30901604613834, 143.31601124724918, 143.31236344298893, 143.3107755752521, 143.25589646028465, 143.25589109586662]


# In[13]:"""


"""if __name__ == "__main__":
    spatial_map,final_map = final_maps_api_parallel(lat_boundaries,long_boundaries,api_keys)"""


# In[ ]:





# In[14]:


"""
API key for the python programm:
http://127.0.0.1:5000/pollination/?lat_boundaries=-35.08491940916005,-35.11377165513988,-35.11377165513988,-35.128093335964095,-35.12816353408006,-35.14255287012573,-35.14248268441524,-35.12876021562264,-35.11377165513988,-35.11363123404099,-35.098464331083484,-35.08475718058806,-35.08478375410176,-35.08486825609331&long_boundaries=143.2558918258238,143.2560585113483,143.28189354857975,143.28193646392398,143.29060536346012,143.29051953277164,143.3088443847614,143.3087585540729,143.30901604613834,143.31601124724918,143.31236344298893,143.3107755752521,143.25589646028465,143.25589109586662
"""

