import pandas as pd
from pyproj import Transformer
import numpy as np


def preprocess_data(filePath, utmZone="50N"):
    df = pd.read_json(filePath)
    zoneNumber = utmZone[:-1]
    hemisphere = 'north' if utmZone[-1].upper() == 'N' else 'south'
    transformer = Transformer.from_crs("EPSG:4326",
                                       f"+proj=utm +zone={zoneNumber} +{hemisphere} +ellps=WGS84")
    df['location_latitude'] = df['location'].apply(lambda x: x['latitude'])
    df['location_longitude'] = df['location'].apply(lambda x: x['longitude'])
    df['easting'], df['northing'] = transformer.transform(df['location_latitude'].values, df['location_longitude'].values)
    return df, transformer

if __name__ == "__main__":
    dataPath = "../data/Cloud.json"
    # dataPath = "../data/Track.json"
    df, utm_transformer = preprocess_data(dataPath, utmZone='50N')
    for idx, row in df.iterrows():
        print(row['easting'], row['northing'])
    