import os
from jinja2.environment import create_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import folium
# from arcgis.gis import GIS
# from arcgis.geometry import Point
# from arcgis.mapping import MapView
import sqlite3
# from sqlalchemy import create_engine, text

from sklearn.cluster import DBSCAN

#|%%--%%| <vMCDQWSeq0|PVEy53ixSW>
r"""°°°
# Truck Loading and Delivery Routes (TLDR)

Problem Statement:
- Automate assigning customer orders to trucks.
- Optimize truck routes to ensure they arrive at customers via the shortest path.

Data:
- order
- customer
- product
- truck

order:
- Customer # : The unique identifier for each customer.
- Invoice # : The unique identifier for each order.
- Item # : The unique identifier for the product being ordered.
- SalesQty : The quantity of the product ordered.

customer:
- customerNo : The unique identifier for each customer.
- Latitude : The geographical latitude of the customer's location.
- Longitude : The geographical longitude of the customer's location.

product:
- Item # : The unique identifier for the product.
- Product Name : The name of the product.
- Can Size (in mL) : The size of the product can in milliliters.
- Can (in case) : Number of cans per case.
- Packaging (carton) : Details about how the product is packaged in cartons.
- Pallet Size (carton) : The size of the pallet containing cartons.
- Cases per Pallet : The number of cases that fit on a pallet.
- Gross Carton Weight (in Kg) : The weight of the carton in kilograms.
- Carton Length (in cm) : The length of the carton in centimeters.
- Carton Width (in cm) : The width of the carton in centimeters.
- Carton Height (in cm) : The height of the carton in centimeters.

truck:
- Truck ID : The unique identifier for each truck.
- Truck Name : The name or model of the truck.
- Capacity (Pallets) : The truck's carrying capacity in terms of pallets.
- Multiple Trips (# of trips) : The number of trips the truck can make.
- Priority : The priority level assigned to the truck for deliveries.

°°°"""

#|%%--%%| <PVEy53ixSW|2FFA0S85fi>
r"""°°°
## Create Dataframes from csv files
°°°"""
#|%%--%%| <2FFA0S85fi|WXTD6sKCRM>

order = pd.read_csv("Orders.csv")
customer = pd.read_csv("Customer.csv")
product = pd.read_csv("Product.csv")
truck = pd.read_csv("Trucks.csv")


#|%%--%%| <WXTD6sKCRM|S9n94HHdeT>
r"""°°°
## Create Dataframes from csv files
°°°"""
#|%%--%%| <S9n94HHdeT|mjv3LfNFK3>

# Convert all 2nd row to columns for each table
tables = [order, customer, product, truck]

for table in tables:
    table.columns = table.iloc[1]
    # table = table.iloc[2:, :]
    table = table.drop([0,1], axis=0, inplace=True)

#|%%--%%| <mjv3LfNFK3|ltBE3179i6>

for table in tables:
    table.info()
    print("\n")

#|%%--%%| <ltBE3179i6|A7K7x2nzxK>
r"""°°°
### Convert all float text to real float
°°°"""
#|%%--%%| <A7K7x2nzxK|ixu2u4NKf9>
char = "٫"
converter = lambda c: c.replace(char, ".")

for i in [1, 2]:
    customer[customer.columns[i]] = customer[customer.columns[i]].apply(converter).apply(float)

for i in [7, 8, 9, 10]:
    product[product.columns[i]] = product[product.columns[i]].apply(converter).apply(float)



# for i in [7, 8, 10]:
#     product[product.columns[i]] = product[product.columns[i]].apply(converter).apply(float)


#|%%--%%| <ixu2u4NKf9|Vo3wspte9e>

# print(order.head(5))
# print(product.head(5))
# print(customer.head(5))
print(truck.head(5))

#|%%--%%| <Vo3wspte9e|zYK7cQ7lZ2>
r"""°°°
## Create SQLite3 Databse
°°°"""
#|%%--%%| <zYK7cQ7lZ2|ssUu4wsfSr>

def create_sqlite_database(filename):
    """ create a database connection to an SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(filename)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


create_sqlite_database("database.db")

#|%%--%%| <ssUu4wsfSr|yh2wtr04PJ>
r"""°°°
Convert Dataframes to SQLite
docs: 
    - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
    - https://www.sqlitetutorial.net/sqlite-python/creating-database/
°°°"""
#|%%--%%| <yh2wtr04PJ|spFEWUwlny>

# database_file = f'sqlite://{os.getcwd()}/database.db'
# engine = create_engine(database_file, echo=False)


#|%%--%%| <spFEWUwlny|86FIHxXaeQ>

with sqlite3.connect("database.db") as conn:
    customer.to_sql('customer', con=conn, index=False)
    order.to_sql('order', con=conn, index=False)
    product.to_sql('product', con=conn, index=False)
    truck.to_sql('truck', con=conn, index=False)

#|%%--%%| <86FIHxXaeQ|DBygB4kVmR>
with sqlite3.connect("database.db") as conn:
    print(conn.execute("SELECT * FROM product").fetchall())


#|%%--%%| <DBygB4kVmR|L62IfRTi9D>
r"""°°°
### Visualize Customer Locations in grid
°°°"""
target_customer = (21.501795, 39.24419833)
distances = []

for lat, lon in zip(customer['Latitude'], customer['Longitude']):
    distance = geodesic(target_customer, (lat, lon)).kilometers
    distances.append(distance)

# Get the index of the nearest neighbor (excluding the customer itself)
nearest_index = distances.index(sorted(distances)[1])  # [1] to skip the zero distance to itself
nearest_customer_no = customer['customerNo'][nearest_index]

print(f'The nearest customer to 102215 is: {nearest_customer_no}')

#|%%--%%| <L62IfRTi9D|kPtdBso4AJ>
r"""°°°
Visualize Customer Locations in map
°°°"""
#|%%--%%| <kPtdBso4AJ|mutcxhmeUa>


# Create a map centered around the average latitude and longitude
center_lat = sum(customer['Latitude']) / len(customer['Latitude'])
center_lon = sum(customer['Longitude']) / len(customer['Longitude'])
map_customers = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')


#|%%--%%| <mutcxhmeUa|51Wg2rOl7c>

# Add Esri World Imagery basemap
esri_tiles = folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri World Imagery',
    overlay=True,
    control=True
)

esri_tiles.add_to(map_customers)

# Add Google Maps tile layer (optional; this uses a plugin and does not need an API key)
folium.TileLayer('https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}', 
                 attr='Google', name='Google Maps', overlay=True).add_to(map_customers)

# Add layer control
# folium.LayerControl().add_to(map_customers)

# Add customer markers to the map
for lat, lon, cust_no in zip(customer['Latitude'], customer['Longitude'], customer['customerNo']):
    folium.Marker(
        location=[lat, lon],
        popup=f'Customer No: {cust_no}',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_customers)

# Add layer control
folium.LayerControl().add_to(map_customers)

# Save the map to an HTML file
map_customers.save('esri_google_maps_OpenStreetMap_customers.html')

#|%%--%%| <51Wg2rOl7c|SElS4EJ4rP>

# Authenticate with ArcGIS Online (requires an account)
gis = GIS("home")

# Create a map centered around the average latitude and longitude
map_view = gis.map(location=[center_lat, center_lon], zoomlevel=11)

# Add customer locations as graphics
for lat, lon, cust_no in zip(customer_data['Latitude'], customer_data['Longitude'], customer_data['customerNo']):
    point = Point({"x": lon, "y": lat})
    map_view.draw(point, popup={"title": f"Customer No: {cust_no}"})

# Display the map
map_view

#|%%--%%| <SElS4EJ4rP|Ly60dAzsad>
r"""°°°
## K-means with real map
°°°"""
#|%%--%%| <Ly60dAzsad|5PMVvv9hTf>


# Extract latitude and longitude
coordinates = customer[['Latitude', 'Longitude']]

# Perform K-Means clustering
num_clusters = 10  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer['Cluster'] = kmeans.fit_predict(coordinates)

# Create a folium map centered around the average location
center_lat = customer['Latitude'].mean()
center_lon = customer['Longitude'].mean()
map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

# Add Esri World Imagery basemap
esri_tiles = folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri World Imagery',
    overlay=True,
    control=True
)
esri_tiles.add_to(map_clusters)

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen']

# Add customer markers to the map, color-coded by cluster
for _, row in customer.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f'Customer No: {row["customerNo"]}, Cluster: {row["Cluster"]}',
        icon=folium.Icon(color=colors[row['Cluster'] % len(colors)], icon='info-sign')
    ).add_to(map_clusters)

# Add layer control
folium.LayerControl().add_to(map_clusters)

# Save the map to an HTML file
map_clusters.save('customer_clusters_map.html')



#|%%--%%| <5PMVvv9hTf|Ye0EhjOeSw>
r"""°°°
## Customer functions for data 
°°°"""
#|%%--%%| <Ye0EhjOeSw|i6WMGtIZf0>


def get_invoices(customer_id):
    invoices = order[order[order.columns[0]] == customer_id].groupby(order.columns[1])
    return invoices

# len(get_invoices(customer_id))
# len(invoices)

#|%%--%%| <i6WMGtIZf0|RNQ75sf65B>


customer_id = customer[customer.columns[0]].iloc[0]
invoices = get_invoices(customer_id)
invoice_items_len = 0 
for invoice in invoices:
    invoice_items = invoice[1]
    print(invoice_items)
    for item in invoice_items.values:
        qty = item[3]
        print(qty)
        item_data = product[item[2] == product[product.columns[0]]]
        print(item_data.values)
        x = item_data[[product.columns[8], product.columns[9]]].values[0]
        print(x[0], int(x[1]))






#|%%--%%| <RNQ75sf65B|sQI4H6z57Z>
r"""°°°
## DBSCAN with real map
°°°"""
#|%%--%%| <sQI4H6z57Z|YVQJRPqd5K>


def create_row_with_cols_html(key, *args):
    html = f"""<tr><th>{key}</th>"""
    for arg in args:
        html += f"<td>{arg}</td>"
    html += "</tr>"
    return html


def create_row_html(key, value):
    return f"""<tr><th>{key}</th><td>{value}</td></tr>"""


def create_popup_html(row):

    invoices = get_invoices(row["customerNo"])
    inv_len = len(invoices)
    rows = []
    for invoice in invoices:
        invoice_id = invoice[0]
        invoice_items = invoice[1]
        rows.append(create_row_html("invoice_id", invoice_id))
        invoice_area = 0
        for item in invoice_items.values:
            # item_id = item[2]
            # qty = item[3]
            # rows.append(create_row_html("item_id", item_id))
            # rows.append(create_row_html("qty", qty))
            item_data = product[item[2] == product[product.columns[0]]]
            item_l_w = item_data[[product.columns[8], product.columns[9]]].values[0]
            # rows.append(create_row_with_cols_html("Length and Width", *item_l_w))

            invoice_area += item_l_w[0] * item_l_w[1]

        rows.append(create_row_html("Invoice area", invoice_area))



    rows_t = "".join(rows)
    html_content = f"""
    <div style="max-height:200px; overflow-y:auto;">
        <table style="width:250px">
            <tr><th>Customer No</th><td>{row['customerNo']}</td></tr>
            <tr><th>Latitude</th><td>{row['Latitude']}</td></tr>
            <tr><th>Longitude</th><td>{row['Longitude']}</td></tr>
            <tr><th>Invoices counts</th><td>{inv_len}</td></tr>
            {rows_t}
            
        </table>
    </div>
    """
    return folium.Popup(html_content, max_width=300)


# Convert latitude and longitude to radians for haversine distance calculation
coords = np.radians(customer[['Latitude', 'Longitude']])

# Set the maximum distance (in kilometers) for clustering
eps_km = 4  # Adjust this value to set the clustering range in kilometers

# Earth radius in kilometers
earth_radius_km = 6371.0

# Perform DBSCAN clustering
db = DBSCAN(eps=eps_km / earth_radius_km, min_samples=2, metric='haversine').fit(coords)
customer['Cluster'] = db.labels_

# Create a folium map centered around the average location
center_lat = customer['Latitude'].mean()
center_lon = customer['Longitude'].mean()
map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen']

# Add customer markers to the map, color-coded by cluster
for _, row in customer.iterrows():
    cluster_id = row['Cluster']
    color = 'gray' if cluster_id == -1 else colors[cluster_id % len(colors)]  # Use gray for noise points (cluster ID -1)

    # inv_cnts = len(get_invoices(row["customerNo"]))
    # print(list(inv_cnts)[0])

    # popup=f'Customer No: {row["customerNo"]}, <br/> Cluster: {cluster_id}, <br/> inv: {inv_cnts}',
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=create_popup_html(row),
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(map_clusters)


# Add Esri World Imagery basemap
esri_tiles = folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri World Imagery',
    overlay=True,
    control=True
)

esri_tiles.add_to(map_clusters)

# Add Google Maps tile layer (optional; this uses a plugin and does not need an API key)
folium.TileLayer('https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}', 
                 attr='Google', name='Google Maps', overlay=True).add_to(map_clusters)


# Add layer control
folium.LayerControl().add_to(map_clusters)

# Save the map to an HTML file
map_clusters.save('customer_clusters_map_km.html')

#|%%--%%| <YVQJRPqd5K|5u8e3bjDHw>
r"""°°°
## merge data
°°°"""
#|%%--%%| <5u8e3bjDHw|0Ge4TF6mfo>

customer_orders = customer.merge(order, how="inner",  left_on=customer.columns[0], right_on=order.columns[0])
customer_orders.drop(order.columns[0], axis=1, inplace=True)

#|%%--%%| <0Ge4TF6mfo|6InNtycW8y>

customer_orders.merge(
        product, how="inner",
        left_on=customer_orders.columns[4],
        right_on=product.columns[0]
        )

#|%%--%%| <6InNtycW8y|u5xlp5o3Yo>

