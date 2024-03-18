from flask import *
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
import requests
import ast
from Automated_Process import final_maps_api_parallel, api_to_latlong, read_data_from_mesql, distance_matrix, probability_matrix, convert_one_probability, create_mysql_table
from Automated_Process import PDF_PI_TABLE_PREFIX, HIVE_DETAILS_TABLE_PREFIX
from download_weather_data import temporal_processing_time_parallel
import multiprocessing
#in linux deployment uncomment below line
multiprocessing.set_start_method('spawn')

API_KEYS = ["8ee842d65cf08ec205365865e3d53348", "c29f27459329c5bbffb6e633e0fc4502", "51fb82fd74e9c378a1983d2551733418", "94979cc6f8c54c197d859f25576fb942", "21492bb5c90d0b7156cb1c5c543cb3c2"]
FARM_DETAILS_URL = "http://ec2-54-169-248-22.ap-southeast-1.compute.amazonaws.com:5000"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/",  methods=['GET'])

def home():

    try:
        data_set_map = {"map":"status ok"}
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map
    
    except Exception as e:
        data_set_map = {"map":e}
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map


@app.route("/pollination/",  methods=['GET'])

def save_map():

    try:
        lat_boundaries = list(eval(str(request.args.get("lat_boundaries"))))
        long_boundaries = list(eval(str(request.args.get("long_boundaries"))))
        BID = str(request.args.get("business_id"))
        FID = str(request.args.get("farm_id"))

        spatial_html_content,finalmap_html_content = final_maps_api_parallel(lat_boundaries,long_boundaries,API_KEYS,BID,FID)
        
        # if the hivelocaton data is not entered
        if (spatial_html_content==False) and (finalmap_html_content==False):
            data_set_map = {"final_map":"please provide valied hive locations", "spatial_map":"please provide valied hive locations"}
        else:
            data_set_map = {"final_map":finalmap_html_content, "spatial_map":spatial_html_content}
    
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map
        
    except Exception as e:
        error  = str(e)
        error_msg = "'float' object has no attribute 'save'"
        if error == error_msg:
            data_set_map = {"error":"Please choose correct farm boundaries"}
            json_dump_map = json.dumps(data_set_map)
            return json_dump_map
        else:
            data_set_map = {"error":error}
            json_dump_map = json.dumps(data_set_map)
            return json_dump_map

     
@app.route("/timing/",  methods=['GET'])

def processing_time_calculation():    
    try:
        lat_boundaries = list(eval(str(request.args.get("lat_boundaries"))))
        long_boundaries = list(eval(str(request.args.get("long_boundaries"))))

        process_time = temporal_processing_time_parallel(lat_boundaries,long_boundaries)

        message = f"It will takes {process_time}  minutes to Complete!"
        data_set = {"time":process_time, "message":message}
        json_dump = json.dumps(data_set)
        return json_dump
     
    except Exception as e:
        data_set = {"error":str(e)}
        json_dump = json.dumps(data_set)
        return json_dump


@app.route("/update_spatial_probability/",  methods=['GET'])

def update_pdf_pi_table():

    try: 
        BID = str(request.args.get("business_id"))
        FID = str(request.args.get("farm_id"))
        url = f"{FARM_DETAILS_URL}/farm/{FID}"
        HIVE_DETAILS_TABLE = f"{HIVE_DETAILS_TABLE_PREFIX}_{BID}_{FID}"
        PDF_PI_TABLE = f"{PDF_PI_TABLE_PREFIX}_{BID}_{FID}"
        # get farm boundaries
        response = requests.get(url)  
        json_data = response.json()
        # Extract the "boundaries" field from the JSON response
        boundaries_str = json_data.get("boundaries")
        # convert string list as a list of tpples
        boundaries_list = ast.literal_eval(boundaries_str)
        lat_boundaries = [point[0] for point in boundaries_list]
        long_boundaries = [point[1] for point in boundaries_list]

        # get lat long grid points
        lat, long, cols, raws = api_to_latlong(lat_boundaries,long_boundaries)
        # read hive details table
        hive_details_dataset = read_data_from_mesql(HIVE_DETAILS_TABLE)
        # calculate probability distances
        distances = distance_matrix(lat,long,hive_details_dataset)
        porbabilities = probability_matrix(distances,hive_details_dataset)  
        norm_sum_prob_arr = convert_one_probability(porbabilities)
        # update pdf pi table
        data = {'id':np.arange(1,len(norm_sum_prob_arr)+1,1), 'longitude':long,'latitude':lat, 'spatial_prob':norm_sum_prob_arr}
        dataset = pd.DataFrame(data)
        create_mysql_table(dataset, PDF_PI_TABLE)

        message = {"status":"spatial details are sucessfully updated"}
        json_dump = json.dumps(message)
        return json_dump

    except Exception as e:
        message = {"error":str(e)}
        json_dump = json.dumps(message)
        return json_dump



if __name__ == "__main__":
    app.run(debug=True)