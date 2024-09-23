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
    # table.info()
    print(table.isna().sum())
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


order['SalesQty'] = order['SalesQty'].apply(float)
# for i in [7, 8, 10]:
#     product[product.columns[i]] = product[product.columns[i]].apply(converter).apply(float)


#|%%--%%| <ixu2u4NKf9|Vo3wspte9e>

print(order.head(5))
print(product.head(5))
print(customer.head(5))
print(truck.head(5))
  

#|%%--%%| <Vo3wspte9e|FCehHrom8d>
# order.isna().sum()


#|%%--%%| <FCehHrom8d|zYK7cQ7lZ2>
r"""°°°
## Create SQLite3 Databse
°°°"""
#|%%--%%| <zYK7cQ7lZ2|ssUu4wsfSr>


# def create_sqlite_database(filename):
#     """ create a database connection to an SQLite database """
#     conn = None
#     try:
#         conn = sqlite3.connect(filename)
#         print(sqlite3.sqlite_version)
#     except sqlite3.Error as e:
#         print(e)
#     finally:
#         if conn:
#             conn.close()


# create_sqlite_database("database.db")

#|%%--%%| <ssUu4wsfSr|yh2wtr04PJ>
r"""°°°
### Convert Dataframes to SQLite
docs: 
    - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
    - https://www.sqlitetutorial.net/sqlite-python/creating-database/
°°°"""
#|%%--%%| <yh2wtr04PJ|spFEWUwlny>

# database_file = f'sqlite://{os.getcwd()}/database.db'
# engine = create_engine(database_file, echo=False)


#|%%--%%| <spFEWUwlny|86FIHxXaeQ>

# with sqlite3.connect("database.db") as conn:
#     customer.to_sql('customer', con=conn, index=False)
#     order.to_sql('order', con=conn, index=False)
#     product.to_sql('product', con=conn, index=False)
#     truck.to_sql('truck', con=conn, index=False)

#|%%--%%| <86FIHxXaeQ|DBygB4kVmR>
# with sqlite3.connect("database.db") as conn:
#     print(conn.execute("SELECT * FROM product").fetchall())


#|%%--%%| <DBygB4kVmR|L62IfRTi9D>
r"""°°°
### Visualize Customer Locations in grid
°°°"""

#|%%--%%| <L62IfRTi9D|2w0UmeNkdd>
# target_customer = (21.501795, 39.24419833)
# distances = []

# for lat, lon in zip(customer['Latitude'], customer['Longitude']):
#     distance = geodesic(target_customer, (lat, lon)).kilometers
#     distances.append(distance)

# # Get the index of the nearest neighbor (excluding the customer itself)
# nearest_index = distances.index(sorted(distances)[1])  # [1] to skip the zero distance to itself
# nearest_customer_no = customer['customerNo'][nearest_index]

# print(f'The nearest customer to 102215 is: {nearest_customer_no}')




#|%%--%%| <2w0UmeNkdd|kPtdBso4AJ>
r"""°°°
Visualize Customer Locations in map
°°°"""
#|%%--%%| <kPtdBso4AJ|mutcxhmeUa>


# # Create a map centered around the average latitude and longitude
# center_lat = sum(customer['Latitude']) / len(customer['Latitude'])
# center_lon = sum(customer['Longitude']) / len(customer['Longitude'])
# map_customers = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')


#|%%--%%| <mutcxhmeUa|51Wg2rOl7c>

# # Add Esri World Imagery basemap
# esri_tiles = folium.TileLayer(
#     tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#     attr='Esri',
#     name='Esri World Imagery',
#     overlay=True,
#     control=True
# )

# esri_tiles.add_to(map_customers)

# # Add Google Maps tile layer (optional; this uses a plugin and does not need an API key)
# folium.TileLayer('https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}', 
#                  attr='Google', name='Google Maps', overlay=True).add_to(map_customers)

# # Add layer control
# # folium.LayerControl().add_to(map_customers)

# # Add customer markers to the map
# for lat, lon, cust_no in zip(customer['Latitude'], customer['Longitude'], customer['customerNo']):
#     folium.Marker(
#         location=[lat, lon],
#         popup=f'Customer No: {cust_no}',
#         icon=folium.Icon(color='blue', icon='info-sign')
#     ).add_to(map_customers)

# # Add layer control
# folium.LayerControl().add_to(map_customers)

# # Save the map to an HTML file
# map_customers.save('esri_google_maps_OpenStreetMap_customers.html')

#|%%--%%| <51Wg2rOl7c|SElS4EJ4rP>

# # Authenticate with ArcGIS Online (requires an account)
# gis = GIS("home")

# # Create a map centered around the average latitude and longitude
# map_view = gis.map(location=[center_lat, center_lon], zoomlevel=11)

# # Add customer locations as graphics
# for lat, lon, cust_no in zip(customer_data['Latitude'], customer_data['Longitude'], customer_data['customerNo']):
#     point = Point({"x": lon, "y": lat})
#     map_view.draw(point, popup={"title": f"Customer No: {cust_no}"})

# # Display the map
# map_view

#|%%--%%| <SElS4EJ4rP|Ly60dAzsad>
r"""°°°
## K-means with real map
°°°"""
#|%%--%%| <Ly60dAzsad|5PMVvv9hTf>


# # Extract latitude and longitude
# coordinates = customer[['Latitude', 'Longitude']]

# # Perform K-Means clustering
# num_clusters = 10  # You can adjust the number of clusters
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# customer['Cluster'] = kmeans.fit_predict(coordinates)

# # Create a folium map centered around the average location
# center_lat = customer['Latitude'].mean()
# center_lon = customer['Longitude'].mean()
# map_clusters = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

# # Add Esri World Imagery basemap
# esri_tiles = folium.TileLayer(
#     tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#     attr='Esri',
#     name='Esri World Imagery',
#     overlay=True,
#     control=True
# )
# esri_tiles.add_to(map_clusters)

# # Define colors for clusters
# colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen']

# # Add customer markers to the map, color-coded by cluster
# for _, row in customer.iterrows():
#     folium.Marker(
#         location=[row['Latitude'], row['Longitude']],
#         popup=f'Customer No: {row["customerNo"]}, Cluster: {row["Cluster"]}',
#         icon=folium.Icon(color=colors[row['Cluster'] % len(colors)], icon='info-sign')
#     ).add_to(map_clusters)

# # Add layer control
# folium.LayerControl().add_to(map_clusters)

# # Save the map to an HTML file
# map_clusters.save('customer_clusters_map.html')



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

# Include necessary CSS for the hover-based dropdown
css = """
<style>
    .invoice-container {
        position: relative;
        display: inline-block;
    }
    .invoice-button {
        padding: 5px 10px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        cursor: pointer;
    }
    .dropdown-menu {
        display: none;
        position: absolute;
        background-color: #f9f9f9;
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 1;
    }
    .dropdown-menu ul {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }
    .dropdown-menu li {
        padding: 8px 12px;
    }
    .invoice-container:hover .dropdown-menu {
        display: block;
    }
</style>
"""



def create_row_html_with_dropdown(key, invoice_id, invoice_items):
    # Create list items for each product in the invoice
    items_html = "".join(
        # f"<li>{item[2]}: {product[product.columns[8]]} x {product[product.columns[9]]}</li>"
        f"<li> : Hello world</li>"
        for item in invoice_items.values
    )
    
    # Construct the dropdown container with hover functionality
    dropdown_html = f"""
    <div class="invoice-container">
        <div class="invoice-button">{invoice_id}</div>
        <div class="dropdown-menu">
            <ul>
                {items_html}
            </ul>
        </div>
    </div>
    """
    

    
    return f"""<tr><th>{key}</th><td>{dropdown_html}</td></tr>"""


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
        fit_on_truck = 0
        for item in invoice_items.values:
            item_data = product[item[2] == product[product.columns[0]]]
            item_l_w = item_data[[product.columns[8], product.columns[9]]].values[0]

            invoice_area += item_l_w[0] * item_l_w[1]
            # fit_on_truck += 12000 / (item_l_w[0] * item_l_w[1])

        rows.append(create_row_html("Invoice area", invoice_area))
        rows.append(create_row_html("Invoice items fit in truck", 12000/invoice_area))
        # rows.append(create_row_html("Invoice items fit in truck # in loop", fit_on_truck))



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

# customer_orders = customer.merge(order, how="inner",  left_on=customer.columns[0], right_on=order.columns[0])
# customer_orders.drop(order.columns[0], axis=1, inplace=True)

#|%%--%%| <0Ge4TF6mfo|6InNtycW8y>

# customer_orders.merge(
#         product, how="inner",
#         left_on=customer_orders.columns[4],
#         right_on=product.columns[0]
#         )

#|%%--%%| <6InNtycW8y|adXK8w89SD>
r"""°°°
## Vehicle Routing Problem (VRP)
°°°"""
#|%%--%%| <adXK8w89SD|rfyXJb6oQP>


# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp
# import numpy as np

# # Sample data: distances between locations (in km) using Euclidean distance
# locations = [
#     (21.501795, 39.24419833),  # Depot
#     (21.74588167, 39.19407),
#     (21.58945167, 39.21912167),
#     (21.45495833, 39.20741167),
#     (21.7710044, 39.2204109),
#     (21.5414813, 39.2886656)
# ]
# depot_index = 0  # Depot is the starting location

# # Vehicle properties
# num_vehicles = 2
# vehicle_capacity = 15  # Example capacity (units of goods)

# # Customer demands (including depot, which has 0 demand)
# demands = [0, 3, 4, 5, 2, 4]

# # Calculate Euclidean distance between locations (distance matrix)
# def compute_euclidean_distance_matrix(locations):
#     distances = np.zeros((len(locations), len(locations)))
#     for i, loc1 in enumerate(locations):
#         for j, loc2 in enumerate(locations):
#             if i != j:
#                 distances[i][j] = np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
#     return distances

# distance_matrix = compute_euclidean_distance_matrix(locations)

# # Create the routing model
# manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_index)
# routing = pywrapcp.RoutingModel(manager)

# # Create and register a transit callback (distance)
# def distance_callback(from_index, to_index):
#     from_node = manager.IndexToNode(from_index)
#     to_node = manager.IndexToNode(to_index)
#     return distance_matrix[from_node][to_node]

# transit_callback_index = routing.RegisterTransitCallback(distance_callback)
# routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# # Add capacity constraint
# def demand_callback(from_index):
#     from_node = manager.IndexToNode(from_index)
#     return demands[from_node]

# demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
# routing.AddDimensionWithVehicleCapacity(
#     demand_callback_index,  # demand callback
#     0,  # null capacity slack
#     [vehicle_capacity] * num_vehicles,  # vehicle maximum capacities
#     True,  # start cumul to zero
#     'Capacity'
# )

# # Set search parameters
# search_parameters = pywrapcp.DefaultRoutingSearchParameters()
# search_parameters.first_solution_strategy = (
#     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# # Solve the problem
# solution = routing.SolveWithParameters(search_parameters)

# # Print the solution
# if solution:
#     for vehicle_id in range(num_vehicles):
#         index = routing.Start(vehicle_id)
#         route_distance = 0
#         route_load = 0
#         route = []
#         while not routing.IsEnd(index):
#             node_index = manager.IndexToNode(index)
#             route_load += demands[node_index]
#             route.append(node_index)
#             previous_index = index
#             index = solution.Value(routing.NextVar(index))
#             route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
#         route.append(manager.IndexToNode(index))
#         print(f"Route for vehicle {vehicle_id}: {route}")
#         print(f"Distance of the route: {route_distance}")
#         print(f"Load of the route: {route_load}")
# else:
#     print('No solution found!')

#|%%--%%| <rfyXJb6oQP|UHZSn3DdRN>


# Aggregate demand for each customer
customer_demand = order.groupby('Customer #')['SalesQty'].sum().reset_index()
customer_demand.columns = ['customerNo', 'demand']
print(customer_demand.head())

#|%%--%%| <UHZSn3DdRN|XVJBr6P2Qk>


# Merge customer demand with geolocation data
customer_data = pd.merge(customer, customer_demand, on='customerNo', how='inner')
print(customer_data.head())

#|%%--%%| <XVJBr6P2Qk|u5xlp5o3Yo>

# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp
# import numpy as np

# # Extract customer locations (including a depot)
# depot = (21.501795, 39.244198)  # Replace with actual depot location
# customer_locations = customer_data[['Latitude', 'Longitude']].values.tolist()
# locations = [depot] + customer_locations  # Depot + all customer locations

# # Vehicle properties
# num_vehicles = len(truck)  # Number of vehicles
# vehicle_capacities = truck['Capacity (Pallets)'].astype(int).tolist()

# # Customer demands (including depot with 0 demand)
# demands = [0] + customer_data['demand'].tolist()

# # Calculate Euclidean distance between locations (distance matrix)
# def compute_euclidean_distance_matrix(locations):
#     distances = np.zeros((len(locations), len(locations)))
#     for i, loc1 in enumerate(locations):
#         for j, loc2 in enumerate(locations):
#             if i != j:
#                 distances[i][j] = np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
#     return distances


# distance_matrix = compute_euclidean_distance_matrix(locations)

# # Create the routing model
# manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)  # Depot index is 0
# routing = pywrapcp.RoutingModel(manager)


# # Create and register a transit callback (distance)
# def distance_callback(from_index, to_index):
#     from_node = manager.IndexToNode(from_index)
#     to_node = manager.IndexToNode(to_index)
#     return distance_matrix[from_node][to_node]


# transit_callback_index = routing.RegisterTransitCallback(distance_callback)
# routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


# # Add capacity constraint
# def demand_callback(from_index):
#     from_node = manager.IndexToNode(from_index)
#     return demands[from_node]


# demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
# routing.AddDimensionWithVehicleCapacity(
#     demand_callback_index,  # demand callback
#     0,  # null capacity slack
#     vehicle_capacities,  # vehicle capacities
#     True,  # start cumul to zero
#     'Capacity'
# )

# # Set search parameters
# search_parameters = pywrapcp.DefaultRoutingSearchParameters()
# search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

# # Solve the problem
# solution = routing.SolveWithParameters(search_parameters)

# # Print the solution
# if solution:
#     for vehicle_id in range(num_vehicles):
#         index = routing.Start(vehicle_id)
#         route_distance = 0
#         route_load = 0
#         route = []
#         while not routing.IsEnd(index):
#             node_index = manager.IndexToNode(index)
#             route_load += demands[node_index]
#             route.append(node_index)
#             previous_index = index
#             index = solution.Value(routing.NextVar(index))
#             route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
#         route.append(manager.IndexToNode(index))
#         print(f"Route for vehicle {vehicle_id}: {route}")
#         print(f"Distance of the route: {route_distance}")
#         print(f"Load of the route: {route_load}")
# else:
#     print('No solution found!')

#|%%--%%| <u5xlp5o3Yo|KIjq3S4sOL>



# import folium
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp
# import pandas as pd
# import numpy as np

# # Assuming `customer_data`, `truck`, and other variables are defined as in the previous example

# # Depot location (latitude, longitude)
# depot = (21.501795, 39.244198)  # Replace with actual depot coordinates
# locations = [depot] + customer[['Latitude', 'Longitude']].values.tolist()

# # Calculate Euclidean distance between locations (distance matrix)
# def compute_euclidean_distance_matrix(locations):
#     distances = np.zeros((len(locations), len(locations)))
#     for i, loc1 in enumerate(locations):
#         for j, loc2 in enumerate(locations):
#             if i != j:
#                 distances[i][j] = np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
#     return distances

# distance_matrix = compute_euclidean_distance_matrix(locations)

# # Vehicle properties
# num_vehicles = len(truck)
# vehicle_capacities = truck['Capacity (Pallets)'].astype(int).tolist()
# demands = [0] + customer['demand'].tolist()

# # Create the routing model
# manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
# routing = pywrapcp.RoutingModel(manager)

# # Define the distance callback
# def distance_callback(from_index, to_index):
#     from_node = manager.IndexToNode(from_index)
#     to_node = manager.IndexToNode(to_index)
#     return distance_matrix[from_node][to_node]

# transit_callback_index = routing.RegisterTransitCallback(distance_callback)
# routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# # Add capacity constraint
# def demand_callback(from_index):
#     from_node = manager.IndexToNode(from_index)
#     return demands[from_node]

# demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
# routing.AddDimensionWithVehicleCapacity(
#     demand_callback_index,
#     0,
#     vehicle_capacities,
#     True,
#     'Capacity'
# )

# # Set search parameters
# search_parameters = pywrapcp.DefaultRoutingSearchParameters()
# search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

# # Solve the problem
# solution = routing.SolveWithParameters(search_parameters)

# # Initialize a map centered at the depot
# m = folium.Map(location=depot, zoom_start=12)

# # Add depot marker
# folium.Marker(depot, popup='Depot', icon=folium.Icon(color='blue', icon='home')).add_to(m)

# # Colors for routes
# colors = ['red', 'green', 'purple', 'orange', 'blue', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

# # Extract and plot routes
# if solution:
#     for vehicle_id in range(num_vehicles):
#         index = routing.Start(vehicle_id)
#         route = []
#         while not routing.IsEnd(index):
#             node_index = manager.IndexToNode(index)
#             route.append(node_index)
#             index = solution.Value(routing.NextVar(index))
#         route.append(manager.IndexToNode(index))  # Add the depot at the end

#         # Draw the route on the map
#         route_coords = [locations[i] for i in route]
#         folium.PolyLine(route_coords, color=colors[vehicle_id % len(colors)], weight=2.5, opacity=1).add_to(m)

#         # Add markers for each customer in the route
#         for i in route[1:]:  # Skip the first index as it's the depot
#             folium.Marker(
#                 location=locations[i],
#                 popup=f'Customer {customer_data.iloc[i-1]["customerNo"]}, Demand: {demands[i]}',
#                 icon=folium.Icon(color='green', icon='info-sign')
#             ).add_to(m)
# else:
#     print('No solution found!')

# # Save the map to an HTML file and display it
# m.save('vrp_solution_map.html')
#|%%--%%| <KIjq3S4sOL|EwohGJPppP>


# import folium
# import openrouteservice
# import pandas as pd

# # Load your data
# customer_data = customer
# truck_data = truck

# # Your ORS API key (replace 'YOUR_API_KEY' with your real OpenRouteService API key)
# client = openrouteservice.Client(key='5b3ce3597851110001cf6248cb8b884a54184dde970cac3e7a838150')

# # Depot location (latitude, longitude)
# depot = (21.501795, 39.244198)

# # Get customer locations
# customer_locations = customer_data[['Latitude', 'Longitude']].values.tolist()

# # Initialize a map centered at the depot
# m = folium.Map(location=depot, zoom_start=12)

# # Add depot marker
# folium.Marker(depot, popup='Depot', icon=folium.Icon(color='blue', icon='home')).add_to(m)

# # Add markers for each customer
# for i, (lat, lon) in enumerate(customer_locations):
#     folium.Marker(
#         location=(lat, lon),
#         popup=f'Customer {customer_data.iloc[i]["customerNo"]}',
#         icon=folium.Icon(color='green', icon='info-sign')
#     ).add_to(m)

# # # Construct a route with ORS between the depot and each customer
# # for _customer in customer_locations[:5]:
# #     coords = [depot, _customer]  # Coordinates for the route: from depot to customer

# #     # Make a request to the ORS API for directions
# #     try:
# #         route = client.directions(
# #             coordinates=coords,
# #             profile='driving-car',
# #             format='geojson'
# #         )
# #         # Extract the geometry of the route
# #         folium.GeoJson(route['routes'][0]['geometry'], name='route').add_to(m)
# #     except Exception as e:
# #         print(f"An error occurred while fetching the route: {e}")



# # Construct a route with ORS between the depot and each customer
# # for _customer in customer_locations[:5]:
# #     coords = [depot, _customer]  # Coordinates for the route: from depot to customer

# #     # Make a request to the ORS API for directions with an increased radius
# #     try:
# #         route = client.directions(
# #             coordinates=coords,
# #             profile='driving-car',
# #             format='geojson',
# #             radiuses=[1000, 1000]  # Increase search radius to 1000 meters for both points
# #         )
# #         # Extract the geometry of the route
# #         folium.GeoJson(route['routes'][0]['geometry'], name='route').add_to(m)
# #     except Exception as e:
# #         print(f"An error occurred while fetching the route: {e}")
# # Construct a route with ORS between the depot and each customer
# for _customer in customer_locations[:]:
#     coords = [depot, _customer]  # Coordinates for the route: from depot to customer

#     try:
#         # Request directions from ORS API
#         route = client.directions(
#             coordinates=coords,
#             profile='driving-car',
#             format='geojson',
#             radiuses=[1000, 1000]  # Increase search radius
#         )

#         # Check if 'routes' is in the response
#         if 'routes' in route:
#             # Extract the geometry of the route
#             folium.GeoJson(route['routes'][0]['geometry'], name='route').add_to(m)
#         else:
#             print(f"Routing failed for coordinates: {coords}")
#             print(f"Full response from ORS: {route}")

#     except Exception as e:
#         print(f"An error occurred while fetching the route: {e}")


# # Save the map to an HTML file
# m.save('realistic_vrp_map.html')
# # m

