from flask import *
from flask_cors import CORS
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow 
from sqlalchemy import PrimaryKeyConstraint
from flask import request
import json
import pandas as pd
import numpy as np
import requests
import ast
from datetime import datetime
from Automated_Process import final_maps_api_parallel, api_to_latlong, read_data_from_mesql, distance_matrix, probability_matrix, convert_one_probability, create_mysql_table
from Automated_Process import PDF_PI_TABLE_PREFIX, HIVE_DETAILS_TABLE_PREFIX
from download_weather_data import temporal_processing_time_parallel
import multiprocessing
#in linux deployment uncomment below line
multiprocessing.set_start_method('spawn')

API_KEYS = ["8ee842d65cf08ec205365865e3d53348", "c29f27459329c5bbffb6e633e0fc4502", "51fb82fd74e9c378a1983d2551733418", "94979cc6f8c54c197d859f25576fb942", "21492bb5c90d0b7156cb1c5c543cb3c2"]
FARM_DETAILS_URL = "http://ec2-54-169-248-22.ap-southeast-1.compute.amazonaws.com:5000"
WEATHER_DETAILS_ORDER = [
        "id",
        "date",
        "time",
        "longitude",
        "latitude",
        "tempreture",
        "humidity",
        "wind_speed",
        "weather_id",
        "weather_id_group",
        "weather_id_description",
        "sunrise",
        "sunset",
        "weather_condition_prob",
        "hour_prob",
        "tempreture_prob",
        "humidity_prob",
        "wind_prob",
        "weather_prob",
        "spatial_prob",
        "final_prob"
]

WEATHER_UNITS = [{
    "tempreture": "Celsius",
    "humidity": "%",
    "wind_speed": "meter/sec"
}]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
db = SQLAlchemy()
ma = Marshmallow()

# MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://dilshan:1234@localhost/broodbox'
db.init_app(app)

# Weather details class
# Global dictionary to store dynamically created Weather classes
dynamic_weather_classes = {}

# Weather details class
def create_weather_class(business_id, farm_id):
    table_name = f"final_weather_data_{business_id}_{farm_id}"
    
    # Check if the Weather class for this table already exists
    if table_name in dynamic_weather_classes:
        Weather = dynamic_weather_classes[table_name]
    else:
        class Weather(db.Model): 
            __tablename__ = table_name
            
            time = db.Column(db.String(255), nullable=False)
            date =  db.Column(db.String(255), nullable=False)
            longitude = db.Column(db.Double, nullable=False)
            latitude = db.Column(db.Double, nullable=False)
            tempreture = db.Column(db.Float, nullable=False)
            humidity = db.Column(db.Float, nullable=False)
            wind_speed = db.Column(db.Float, nullable=False)
            weather_id = db.Column(db.Integer, nullable=False)
            weather_id_group = db.Column(db.String(255), nullable=False)
            weather_id_description = db.Column(db.String(255), nullable=False)
            sunrise = db.Column(db.Date, nullable=False)
            sunset = db.Column(db.Date, nullable=False)
            id = db.Column(db.Integer, nullable=False)
            weather_condition_prob = db.Column(db.Float, nullable=False)
            hour_prob =  db.Column(db.Integer, nullable=False)
            tempreture_prob = db.Column(db.Float, nullable=False)
            humidity_prob = db.Column(db.Float, nullable=False)
            wind_prob = db.Column(db.Float, nullable=False)
            weather_prob = db.Column(db.Float, nullable=False)
            spatial_prob = db.Column(db.Float, nullable=False)
            final_prob = db.Column(db.Float, nullable=False)
            mask = db.Column(db.Integer, nullable=False)


            __table_args__ = (
                PrimaryKeyConstraint('id'),
            )

            def __init__(self, time, date, longitude, latitude, tempreture, humidity, wind_speed, weather_id, weather_id_group, weather_id_description, sunrise, sunset, id, 
                         weather_condition_prob, hour_prob, tempreture_prob, humidity_prob, wind_prob, weather_prob, spatial_prob, final_prob, mask):

                self.time = time
                self.date = date 
                self.longitude = longitude
                self.latitude = latitude
                self.tempreture = tempreture
                self.humidity = humidity
                self.wind_speed = wind_speed 
                self.weather_id = weather_id
                self.weather_id_group = weather_id_group
                self.weather_id_description = weather_id_description
                self.sunrise = sunrise
                self.sunset = sunset
                self.id = id
                self.weather_condition_prob = weather_condition_prob
                self.hour_prob = hour_prob
                self.tempreture_prob = tempreture_prob
                self.humidity_prob = humidity_prob
                self.wind_prob = wind_prob
                self.weather_prob = weather_prob
                self.spatial_prob = spatial_prob
                self.final_prob = final_prob
                self.mask = mask

            
        # Save the class in the global dictionary
        dynamic_weather_classes[table_name] = Weather

    return Weather


class WeatherSchema(ma.Schema):
    class Meta:
        fields = ('time', 'date', 'longitude', 'latitude', 'tempreture', 'humidity', 'wind_speed', 'weather_id', 'weather_id_group', 'weather_id_description', 'sunrise', 'sunset', 
                  'id', 'weather_condition_prob', 'hour_prob', 'tempreture_prob', 'humidity_prob', 'wind_prob', 'weather_prob', 'spatial_prob', 'final_prob')

Weather_schema = WeatherSchema()
Weathers_schema  = WeatherSchema(many=True)

# Maps details class
# Global dictionary to store dynamically created Maps classes
dynamic_maps_classes = {}

# Maps details class
def create_map_class(business_id, farm_id):
    table_name = f"maps_{business_id}_{farm_id}"
    
    # Check if the Maps class for this table already exists
    if table_name in dynamic_maps_classes:
        Map = dynamic_maps_classes[table_name]
    else:
        class Map(db.Model): 
            __tablename__ = table_name
            
            id = db.Column(db.Integer, nullable=False)         
            date =  db.Column(db.Date, nullable=False)
            time = db.Column(db.Time, nullable=False)
            spatial_map = db.Column(db.String, nullable=False)
            final_map = db.Column(db.String, nullable=False)


            __table_args__ = (
                PrimaryKeyConstraint('id'),
            )

            def __init__(self, id, date, time, spatial_map, final_map):
                self.id = id
                self.date = date
                self.time = time
                self.spatial_map = spatial_map
                self.final_map = final_map

        # Save the class in the global dictionary
        dynamic_maps_classes[table_name] = Map

    return Map


class MapsSchema(ma.Schema):
    class Meta:
        fields = ('id', 'date', 'time', 'spatial_map', 'final_map')

Map_schema = MapsSchema()
Maps_schema  = MapsSchema(many=True)

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
        time_zone = str(request.args.get("time_zone"))

        spatial_html_content,finalmap_html_content,map_id = final_maps_api_parallel(lat_boundaries,long_boundaries,API_KEYS,BID,FID,time_zone)
        
        # if the hivelocaton data is not entered
        if (spatial_html_content==False) and (finalmap_html_content==False):
            data_set_map = {"final_map":"please provide valied hive locations", "spatial_map":"please provide valied hive locations"}
        else:
            data_set_map = {"map_id":int(map_id), "final_map":finalmap_html_content, "spatial_map":spatial_html_content}
    
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



# maps returning function
@app.route('/map/get_specific/<business_id>/<farm_id>/<int:current_id>', methods=['GET'])
def get_specific_map(business_id, farm_id,current_id):
    try:
        # if direction is 1 then select the next map. if direction is -1 then select the previous map 
        direction = int(request.args.get('direction', -1)) 
        Map = create_map_class(business_id, farm_id)
        if direction== -1:
            map_result = Map.query.filter(Map.id < current_id).order_by(Map.id.desc()).first()

        if direction== 1:
            map_result = Map.query.filter(Map.id > current_id).order_by(Map.id.asc()).first()
        
        if not map_result:
            return jsonify({"error": "map not found"})

        map_data = Map_schema.dump(map_result)
        return jsonify(map_data)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/map/get_maps_by_date/<business_id>/<farm_id>', methods=['GET'])
def get_maps_on_date(business_id, farm_id):
    try:  
        # Get the date from the request query parameter (format: YYYY-MM-DD)
        target_date_str = request.args.get('date')
        
        # Parse the target date string into a datetime object
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d') if target_date_str else None
        
        Map = create_map_class(business_id, farm_id)
        
        # Filter maps based on the provided date and direction
        map_result = Map.query.filter(Map.date == target_date).all()
 
        if not map_result:
            return jsonify({"error": "map not found"})

        map_data = Maps_schema.dump(map_result)
        return jsonify(map_data)

    except Exception as e:
        return jsonify({"error": str(e)})



# weather dataset returning function
@app.route('/weather/get_all/<business_id>/<farm_id>', methods=['GET'])
def get_all_weathers(business_id, farm_id):
    try:

        Weather = create_weather_class(business_id, farm_id)
        weather = Weather.query.all()

        if not weather:
            return jsonify({"error": "location not found"})

        weather_data = Weathers_schema.dump(weather)
        sorted_weather_data = sorted(weather_data, key=lambda x: x['id'])

        return jsonify([{"column_order":WEATHER_DETAILS_ORDER, "weather_units":WEATHER_UNITS, "data_raws":sorted_weather_data}])

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(debug=True)