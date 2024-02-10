from flask import *
from flask_cors import CORS
import json
from Automated_Process import final_maps_api_parallel
from download_weather_data import temporal_processing_time_parallel
import multiprocessing
#in linux deployment uncomment below line
#multiprocessing.set_start_method('spawn')

API_KEYS = ["8ee842d65cf08ec205365865e3d53348", "c29f27459329c5bbffb6e633e0fc4502", "51fb82fd74e9c378a1983d2551733418", "94979cc6f8c54c197d859f25576fb942", "21492bb5c90d0b7156cb1c5c543cb3c2"]

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

    lat_boundaries = list(eval(str(request.args.get("lat_boundaries"))))
    long_boundaries = list(eval(str(request.args.get("long_boundaries"))))

    spatial_html_content,finalmap_html_content = final_maps_api_parallel(lat_boundaries,long_boundaries,API_KEYS)
       
        # if the hivelocaton data is not entered
    if (spatial_html_content==False) and (finalmap_html_content==False):
        data_set_map = {"final_map":"please provide valied hive locations", "spatial_map":"please provide valied hive locations"}
    else:
        data_set_map = {"final_map":finalmap_html_content, "spatial_map":spatial_html_content}
 
    json_dump_map = json.dumps(data_set_map)
    return json_dump_map
    
"""    except Exception as e:
        data_set_map = {"final_map":str(e), "spatial_map":"Server error please try again"}
        json_dump_map = json.dumps(data_set_map)
        return json_dump_map
"""
        
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
        data_set = {"time":str(e), "message":"Server error please try again"}
        json_dump = json.dumps(data_set)
        return json_dump
    

if __name__ == "__main__":
    app.run(debug=True)