import argparse
import logging
import pandas as pd
import geopandas as gpd
from shapely import wkt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a uniform grid of points over a given area."
    )
    parser.add_argument(
        "-s", "--speed", type=float, required=True, help="Speed value to assign to all edges."
    )
    args = parser.parse_args()

    logging.info(f"Setting all edge speeds to {args.speed} km/h")

    df = pd.read_csv("./edges.csv", sep=";")
    df["geometry"] = df["geometry"].apply(wkt.loads)  # only if geometry is in WKT
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf.set_crs(epsg=4326, inplace=True)  # set CRS if not already set
    gdf["maxspeed"] = args.speed
    
    # Write a geojson file with the updated speeds
    gdf.to_file(f"./edges_{int(args.speed)}.geojson", driver="GeoJSON")
    logging.info(f"Written updated edges to ./edges_{int(args.speed)}.json")