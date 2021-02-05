import skmob
from skmob.preprocessing import filtering, compression, detection, clustering
import psycopg2 as pg
from pymongo import MongoClient 
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from itertools import chain
import datetime
import warnings
warnings.filterwarnings(action='ignore')

def get_date_list(start_date, end_date):
    date_list = [start_date]

    while start_date < end_date:
        start_date += datetime.timedelta(days=1)
        date_list.append(start_date)
    return date_list

def split_df(df, date_list, reset_index=True):
    df_list = [df[df.devicetime < date_list[0]]]

    for idx, _ in enumerate(date_list):
        if idx != (len(date_list) - 1):
            _df = df[(df.devicetime >= date_list[idx]) & (df.devicetime < date_list[idx+1])]
            _df = _df.reset_index(drop=reset_index)
            df_list.append(_df)
    return df_list

def df_to_tdf(df):
    tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude',
                              datetime='devicetime', user_id='uniqueid')
    return tdf

def create_cluster(tdf):
    ftdf = filtering.filter(tdf, max_speed_kmh=300) # Filtering
    ctdf = compression.compress(ftdf, spatial_radius_km=0.2) # Compression

    stdf = detection.stops(ctdf, stop_radius_factor=0.5, minutes_for_a_stop=30,
                           spatial_radius_km=0.2, leaving_time=True)
    cstdf = clustering.cluster(stdf, cluster_radius_km=0.1, min_samples=1)
    return cstdf

def merge(cluster):
    stop_list = list()

    for column in cluster.columns:
        if column == 'leaving_datetime':
            stop_list.append(cluster[column].iloc[-1])
        elif column == 'lat':
            stop_list.append(np.mean(cluster.lat))
        elif column == 'lng':
            stop_list.append(np.mean(cluster.lng))
        else:
            stop_list.append(cluster[column].iloc[0])
    
    df = pd.DataFrame([stop_list], columns=cluster.columns)
    return df

def set_cluster_index(cstdf):
    cluster_idx = 0
    cluster_idx_list = [cluster_idx]
    
    for i in range(len(cstdf) - 1):
        if cstdf.cluster[i+1] != cstdf.cluster[i]:
            cluster_idx += 1
        cluster_idx_list.append(cluster_idx)
    cstdf['cluster_idx'] = cluster_idx_list
    return cstdf

def create_stay_point(cstdf):
    stay_point = pd.DataFrame()
    cstdf = set_cluster_index(cstdf)
    
    for i in cstdf.cluster_idx.unique():
        cluster = cstdf[cstdf.cluster_idx == i]

        if cluster.shape[0] > 1:
            stay_point = stay_point.append(merge(cluster))
        else:
            stay_point = stay_point.append(cluster)
    return stay_point

if __name__ == "__main__":
    print('Running...')
    # NOTICE: traccar(pg) timestamp -- localtime, mongo timestamp -- UTC
    engine = create_engine('postgresql://dblab:dblab4458@166.104.110.153/traccar')
    client = MongoClient('mongodb://dblab:dblab4458!@166.104.110.153:54333/jauntUser')

    sql = "\
        SELECT tp.*, td.name, td.uniqueid, td.attributes as properties\
        FROM tc_positions tp, tc_devices td \
        WHERE NOT(td.name LIKE '@%%' OR td.name LIKE '#%%' OR td.name = '01' OR td.name = '02' OR td.name = 'test') AND (tp.deviceid = td.id) \
        ORDER BY devicetime ASC"

    print('Reading data...')
    df = pd.read_sql(sql, engine)
    df = df[df.devicetime < datetime.datetime(2019, 12, 23, 16, 0)].reset_index(drop=True)
    
    print('Clustering...')
    stay_point = pd.DataFrame()

    for idx, user_id in enumerate(df.uniqueid.unique()):
        user = df[df.uniqueid == user_id].sort_values(by=['devicetime']).reset_index(drop=True)
        
        start_date = datetime.datetime.combine(user.devicetime.iloc[0].date(), datetime.time(20, 0))
        end_date = datetime.datetime.combine(user.devicetime.iloc[-1].date(), datetime.time())
        
        date_list = get_date_list(start_date, end_date)
        df_list = split_df(user, date_list, reset_index=True)
        
        for i in df_list:
            try: 
                tdf = df_to_tdf(i)
                cstdf = create_cluster(tdf)
                stay_point = stay_point.append(create_stay_point(cstdf))
            except:
                pass
        print(f'{idx+1}/{len(df.uniqueid.unique())}\r', end='')
    print()

    # Cross-user history data based travel survey
    db = client['jauntUser']
    collection = db['device']

    sql = "select * from tc_devices where not(name like '@%%' or name like '#%%' or name = '01' or name = '02')"
    real_user = pd.read_sql(sql, engine)

    stop_data = db['device'].find({
        "uniqueid" : {'$in': real_user['uniqueid'].to_list()}
    })

    all_stops = []

    for user in list(stop_data):    
        all_stops.append(user['recentStops'])

    stop_df = pd.DataFrame(chain.from_iterable(all_stops))

    print('Validation...')
    for i in stay_point.uid.unique():
        val_1 = stay_point[stay_point.uid == i].shape[0]
        val_2 = stop_df[stop_df.uid == i].shape[0]
        
        if val_1 != val_2:
            print(i, val_1 - val_2)

    print('Saving...')
    stay_point.to_csv('./stay_point.csv', index=False, encoding='utf-8')