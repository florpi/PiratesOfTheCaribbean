import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray


def neighbours_within_radius(gpd_df, cpt, radius):
  """
  :param gpd_df: Geopandas dataframe in which to search for points
  :param cpt:    Point about which to search for neighbouring points
  :param radius: Radius about which to search for neighbours
  :return:       List of point indices around the central point, sorted by
                 distance in ascending order
  """
  #Spatial index
  sindex = gpd_df.sindex
  #Bounding box of rtree search (West, South, East, North)
  bbox = (cpt.x-radius, cpt.y-radius, cpt.x+radius, cpt.y+radius)
  #Potential neighbours
  good = []
  for n in sindex.intersection(bbox):
    dist = cpt.distance(gpd_df['geometry'][n])
    if dist<radius:
      n = gpd_df.loc[n,'id']
      good.append((dist,n))
  #Sort list in ascending order by `dist`, then `n`
  good.sort() 
  #Return only the neighbour indices, sorted by distance in ascending order
  return [x[1] for x in good]


def compute_all_neighbours(gpd_df, radius, probabilities):
    idx_neighbors = []
    num_neighbors = []
    for i, row in tqdm(gpd_df.iterrows(), total=gpd_df.shape[0]):
        cpt = gpd_df.loc[i,'geometry'].centroid
        neighbours = neighbours_within_radius(gpd_df,cpt,radius)
        idx_neighbors.append(neighbours)
        num_neighbors.append(len(neighbours))
    gpd_df[f"idx_neighbors_{radius}"] = idx_neighbors
    gpd_df[f"num_neighbors_{radius}"]  = num_neighbors
    gpd_df[f"mean_area_{radius}"] = gpd_df[f"idx_neighbors_{radius}"].apply(lambda x:gpd_df.loc[gpd_df.id.isin(x), "geometry"].area.mean()).copy()
    #for metal_type in probabilities.columns[1:]:
    #    gpd_df[f"{metal_type}_{radius}"] = gpd_df[f"idx_neighbors_{radius}"].apply(lambda x: probabilities.loc[probabilities.id.isin(x), metal_type].mean())
    return gpd_df

def compute_geometric_features(geojsons, probabilities):
    dfs = []
    radius = [20, 50,100,200]
    zone_to_crs = {'gros_islet': '+init=epsg:32620',
              'castries': '+init=epsg:32620',
              'dennery': '+init=epsg:32620',
              'mixco_3': '+init=epsg:32616',
              'mixco_1_and_ebenezer': '+init=epsg:32616',
              'borde_rural': '+init=epsg:32618',
              'borde_soacha': '+init=epsg:32618'}

    for geojson in geojsons:
        df = gpd.read_file(geojson)
        df["subset"] = "train" if "train" in geojson else "test"
        df["path"] = geojson
        df["zone"] = df["path"].apply(lambda x: x.split("/")[6])
        df = df.to_crs(zone_to_crs[df.at[0,"zone"]])
        for r in radius:
            df = compute_all_neighbours(df, r, probabilities)
        df["area"] = df["geometry"].area.copy()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["place"] = df["path"].apply(lambda x: x.split("/")[5])
    return df

def train_val_split(gdf):
    train_pkl = "drive/My Drive/pirates/data/processed/train/*.pkl"
    val_pkl = "drive/My Drive/pirates/data/processed/validate/*.pkl"
    
    train_idx = pd.concat([pd.read_peakle(train_pkl_ind) for train_pkl_ind in glob.glob(train_pkl)]).element_id
    val_idx = pd.concat([pd.read_peakle(val_pkl_ind) for val_pkl_ind in glob.glob(val_pkl)]).element_id

    return gdf[gdf.element_id.isin(train_idx)].copy(), gdf[gdf.element_id.isin(val_idx)].copy()


