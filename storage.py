import io
import os
import sqlite3

import numpy as np

import config_main

DATABASE_DIR = config_main.DATABASE_DIR
PREFIX = config_main.data['DATABASE_PREFIX']
DATABASE = os.path.join(DATABASE_DIR, PREFIX + '_meatdeli.db')


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)


class DataStorage:
    def __init__(self):
        if not os.path.exists(DATABASE_DIR):
            os.mkdir(DATABASE_DIR)
        if not os.path.exists(DATABASE):
            self.connection = sqlite3.connect(DATABASE, detect_types=1)
            self.create_db()
        else:
            self.connection = sqlite3.connect(DATABASE, detect_types=1)

    def create_db(self):
        c = self.connection.cursor()
        c.execute('''CREATE TABLE hour_data (
            year INTEGER,
            month INTEGER,
            day INTEGER,
            hour INTEGER,
            count INTEGER,
            wait_time INTEGER,
            stay_time INTEGER,
            male INTEGER,
            female INTEGER,
            age_group_1 INTEGER,
            age_group_2 INTEGER,
            age_group_3 INTEGER,
            age_group_4 INTEGER,
            heatmap ARRAY,
            frame ARRAY,
            PRIMARY KEY (year, month, day, hour))''')
        c.execute('''CREATE TABLE customer_data (
            cid INTEGER,
            age REAL,
            gender REAL,
            count INTEGER,
            PRIMARY KEY (cid))''')
        c.execute('''CREATE TABLE visit_data (
            cid INTEGER,
            timestamp REAL,
            PRIMARY KEY (cid, timestamp))''')
        c.execute('''CREATE TABLE heatmap_day (
            year INTEGER,
            month INTEGER,
            day INTEGER,
            heatmap ARRAY,
            PRIMARY KEY (year, month, day))''')
        c.execute('''CREATE TABLE heatmap_month (
            year INTEGER,
            month INTEGER,
            heatmap ARRAY,
            PRIMARY KEY (year, month))''')
        self.connection.commit()

    def insert_into_hour_data(self, year, month, day, hour, data):
        count = data['count']
        wait_time = data['wait_time']
        stay_time = data['stay_time']
        male = data['male']
        female = data['female']
        age1 = data['age1']
        age2 = data['age2']
        age3 = data['age3']
        age4 = data['age4']
        heatmap = data['heatmap']
        frame = data['frame']
        values = (year, month, day, hour, count, wait_time, stay_time, male, female, age1, age2, age3, age4, heatmap, frame)
        c = self.connection.cursor()
        c.execute('INSERT OR REPLACE INTO hour_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)
        self.connection.commit()

    def get_from_hour_data(self, year, month, day, hour):
        c = self.connection.cursor()
        values = (year, month, day, hour)
        c.execute('SELECT * FROM hour_data WHERE year=? AND month=? AND day=? AND hour=?', values)
        data = c.fetchall()
        return data

    def get_day_data(self, year, month, day):
        c = self.connection.cursor()
        values = (year, month, day)
        c.execute('SELECT * FROM hour_data WHERE year=? AND month=? AND day=?', values)
        data = c.fetchall()
        return data

    def get_distinct_years(self):
        c = self.connection.cursor()
        e = c.execute('SELECT DISTINCT(year) FROM hour_data')
        list_years = []
        for row in e:
            list_years.append(row[0])
        return sorted(list_years)

    def get_month_statistic(self, year, month):
        c = self.connection.cursor()
        e = c.execute('SELECT day, SUM(count), SUM(count * wait_time), SUM(count * stay_time), SUM(male), SUM(female), SUM(age_group_1), SUM(age_group_2), SUM(age_group_3), SUM(age_group_4) FROM hour_data WHERE year=? AND month=? GROUP BY day', (year, month))
        return e.fetchall()

    def insert_into_customer_data(self, cid, age, gender, count):
        values = (cid, age, gender, count)
        c = self.connection.cursor()
        c.execute('INSERT OR REPLACE INTO customer_data VALUES (?, ?, ?, ?)', values)
        self.connection.commit()

    def get_from_customer_data(self, cid):
        c = self.connection.cursor()
        values = (cid,)
        c.execute('SELECT * FROM customer_data WHERE cid=?', values)
        data = c.fetchall()
        return data

    def insert_into_visit_data(self, cid, timestamp):
        values = (cid, timestamp)
        c = self.connection.cursor()
        c.execute('INSERT OR REPLACE INTO visit_data VALUES (?, ?)', values)
        self.connection.commit()

    def get_from_visit_data(self, cid):
        c = self.connection.cursor()
        values = (cid,)
        c.execute('SELECT * FROM visit_data WHERE cid=?', values)
        data = c.fetchall()
        return data

    def set_heatmap_day(self, year, month, day, heatmap):
        c = self.connection.cursor()
        values = (year, month, day, heatmap)
        c.execute('INSERT OR REPLACE INTO heatmap_day VALUES (?, ?, ?, ?)', values)
        self.connection.commit()

    def set_heatmap_month(self, year, month, heatmap):
        c = self.connection.cursor()
        values = (year, month, heatmap)
        c.execute('INSERT OR REPLACE INTO heatmap_month VALUES (?, ?, ?)', values)
        self.connection.commit()

    def get_heatmap_day(self, year, month, day):
        c = self.connection.cursor()
        values = (year, month, day)
        c.execute('SELECT heatmap FROM heatmap_day WHERE year=? AND month=? AND day=?', values)
        data = c.fetchall()
        return data

    def get_heatmap_month(self, year, month):
        c = self.connection.cursor()
        values = (year, month)
        c.execute('SELECT heatmap FROM heatmap_month WHERE year=? AND month=?', values)
        data = c.fetchall()
        return data
