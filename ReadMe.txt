1. Create broodbox database

2. create table call 'hive_details' and insert data

3. create table call 'weather_description_map' and insert data

4. install mysql-connector-python
   install sqlalchemy

5. if Lost connection to MySQL server during query error occored
update the xampp my.ini file like including below lines

[mysqld]
max_allowed_packet = 256M
innodb_buffer_pool_size = 512M



AT AWS deployment

1. Create ubuntu 2 GB ram mahcine

2. Install MySQL Server on your Ubuntu instance:
sudo apt update
sudo apt install mysql-server

3. Log in to MySQL:
sudo su
mysql -u root -p
give a password  (12345)

(All other steps are covered in the database API readme file)

4.CREATE TABLE `weather_description_map` (
 `weather_id` int(100), `weather_id_ group` text(255),  `weather_id_description` text(255), 
`mean_ratings` float);


5. clone the git repo
open new terminal 
git clone https://github.com/Chamika-ML/pollination-api-v4.git



6. insert hvie_details.csv and weather_description_map file data to table

go to first
a) SHOW VARIABLES LIKE "secure_file_priv";
   copy  the valu path ( /var/lib/mysql-files/ )

go to second terminal
b) sudo mv /home/ubuntu/pollination-api-v4/data/csv/hive_details.csv /var/lib/mysql-files/
c) sudo mv /home/ubuntu/pollination-api-v4/data/csv/weather_description_map.csv /var/lib/mysql-files/

go to first terminal
d) LOAD DATA INFILE '/var/lib/mysql-files/hive_details.csv'
   INTO TABLE `hive_details_B456_123`
   FIELDS TERMINATED BY ','
   OPTIONALLY ENCLOSED BY '"'
   LINES TERMINATED BY '\n'
   IGNORE 1 LINES;

e) LOAD DATA INFILE '/var/lib/mysql-files/weather_description_map.csv'
   INTO TABLE `weather_description_map`
   FIELDS TERMINATED BY ','
   OPTIONALLY ENCLOSED BY '"'
   LINES TERMINATED BY '\n'
   IGNORE 1 LINES;

7. go the the Automated_process.py and uncomment
MYSQL_CREDENTIALS = {"host":"127.0.0.1", "user":"dilshan", "password":"1234", "database":"broodbox", "port":3306}
and comment other one

8. go to first rerminal 
cd pollination-api-v4
sudo apt-get install python3-venv
python3 -m venv venv2
source venv2/bin/activate
pip install -r requirements.txt

gunicorn -w 4 -b 0.0.0.0:5001 --timeout 0 main:app



