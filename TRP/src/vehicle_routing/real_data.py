from datetime import date, datetime, time, timedelta
from .domain import Location, Visit, Vehicle, VehicleRoutePlan
import pandas as pd
import math
# import os
import sys
# print(sys.path)


SERVICE_DURATION_MINUTES = (10, 20, 30, 40)
MORNING_WINDOW_START = time(8, 0)
MORNING_WINDOW_END = time(12, 0)
AFTERNOON_WINDOW_START = time(13, 0)
AFTERNOON_WINDOW_END = time(18, 0)


prefix_path = sys.path[-1]+"/vehicle_routing/data/"
# prefix_path = "/vehicle_routing/data/"

# Getting data
order = pd.read_csv(prefix_path+"Orders.csv")
customer = pd.read_csv(prefix_path + "Customer.csv")
product = pd.read_csv(prefix_path + "Product.csv")
truck = pd.read_csv(prefix_path + "Trucks.csv")

# Convert all 2nd row to columns for each table
tables = [order, customer, product, truck]

for table in tables:
    table.columns = table.iloc[1]
    table = table.drop([0,1], axis=0, inplace=True)

char = "٫"
converter = lambda c: c.replace(char, ".")

for i in [1, 2]:
    customer[customer.columns[i]] = customer[customer.columns[i]].apply(converter).apply(float)

for i in [7, 8, 9, 10]:
    product[product.columns[i]] = product[product.columns[i]].apply(converter).apply(float)

order['SalesQty'] = order['SalesQty'].apply(int)

for i in [0, 2, 3, 4]:
    truck[truck.columns[i]] = truck[truck.columns[i]].astype(int)


def get_invoices(customer_id):
    invoices = order[order[order.columns[0]] == customer_id].groupby(order.columns[1])
    return invoices


# Assuming `customer` is a DataFrame containing customer details including 'Latitude' and 'Longitude'
customer_locations = customer[['customerNo', 'Latitude', 'Longitude']]

table_rows = []  # List to store the rows for the table

for customer_values in customer.values:
    customer_id = customer_values[0]
    latitude = customer_values[1]
    longitude = customer_values[2]
    
    invoices = get_invoices(customer_id)
    inv_len = len(invoices)
    
    for invoice in invoices:
        invoice_id = invoice[0]
        invoice_items = invoice[1]
        invoice_area = 0
        fit_on_truck = 0
        
        for item in invoice_items.values:
            item_id = item[2]
            qty = int(item[3])
            item_data = product[item[2] == product[product.columns[0]]]
            item_l_w = item_data[[product.columns[8], product.columns[9]]].values[0]
    
            invoice_area += item_l_w[0] * item_l_w[1]
            fit_on_truck += 12000 / (item_l_w[0] * item_l_w[1])
        
        # Append the row data to the table
        table_rows.append({
            "Invoice ID": invoice_id,
            "Fit on Truck": math.floor(fit_on_truck),
            "Latitude": latitude,
            "Longitude": longitude
        })

# Convert to a pandas DataFrame to display as a table
invoice_df = pd.DataFrame(table_rows)



# by customer
demand = 5
latitude = 21.501795
longitude = 39.244198
visits = [Visit(
    id=str(visit_data[0]),
    name=str(visit_data[0]),
    location=Location(latitude=visit_data[1], longitude=visit_data[2]),
    demand=demand,
    min_start_time=datetime.combine(date.today() + timedelta(days=1), MORNING_WINDOW_START),
    max_end_time=datetime.combine(date.today() + timedelta(days=1), AFTERNOON_WINDOW_END),
    service_duration=timedelta(minutes=SERVICE_DURATION_MINUTES[3]),
) for visit_data in customer.values]


 # By using limits and diffrent days
# make partal of invoice if the invoice`s demand total bigger then the limit of turcks and could divide it on the turcks
# visits = []
# limit_trucks = (truck["Capacity (Pallets)"] * truck["Multiple Trips\n(# of trips)"]).sum()
# total_demand = 0
# day_start = 1
# # for visit_data in customer.values:
# for visit_data in invoice_df.head(10).values:
#     visit = Visit(
#             id=str(visit_data[0]),
#             name=str(visit_data[0]),
#             location=Location(latitude=visit_data[2], longitude=visit_data[3]),
#             # demand=demand,
#             demand=visit_data[1],
#             min_start_time=datetime.combine(date.today() + timedelta(days=day_start), MORNING_WINDOW_START),
#             max_end_time=datetime.combine(date.today() + timedelta(days=day_start), AFTERNOON_WINDOW_END),
#             service_duration=timedelta(minutes=SERVICE_DURATION_MINUTES[3]),
#             )
#     # total_demand += demand
#     total_demand += visit_data[1]
#     if total_demand > limit_trucks:
#         # We should divide the product from that invoice but for know will not do it...
#         total_demand = 0
#         day_start += 1
#     elif total_demand == limit_trucks:
#         total_demand = 0
#         day_start += 1
#     visits.append(visit)



# by invoices
# visits = [Visit(
#     id=str(visit_data[0]),
#     name=str(visit_data[0]),
#     location=Location(latitude=visit_data[2], longitude=visit_data[3]),
#     demand=visit_data[1],
#     min_start_time=datetime.combine(date.today() + timedelta(days=1), MORNING_WINDOW_START),
#     max_end_time=datetime.combine(date.today() + timedelta(days=1), AFTERNOON_WINDOW_END),
#     service_duration=timedelta(minutes=SERVICE_DURATION_MINUTES[3]),
# ) for visit_data in table_df.values]


# vehicle capacity, start time, end time
vehicle_start_time = time(7, 30)
# vehicle_end_time = time(15, 30)
vehicle_count = len(truck)
dept_latitude = 21.4339573
dept_longitude = 39.2199178

# vehicle_capacity = 5
vehicles = [Vehicle(id=str(truck_data[1]),
        capacity=truck_data[2],
        home_location=Location(
            latitude=dept_latitude,
            longitude=dept_longitude),
        departure_time=datetime.combine(
            date.today() + timedelta(days=1), vehicle_start_time)
       ) for truck_data in truck.values]

# By using limits and diffrent days
# vehicles = []
# for truck_data in truck.values:
#     for day in range(1, day_start+1):
#         vehicle = Vehicle(id=f"{truck_data[1]} {day} day",
#                           capacity=truck_data[2],
#                           home_location=Location(
#                               latitude=dept_latitude,
#                               longitude=dept_longitude),
#                           departure_time=datetime.combine(
#                               date.today() + timedelta(days=day), vehicle_start_time)
#                           ) 
#         vehicles.append(vehicle)






# center_lat = customer['Latitude'].mean()
# center_lon = customer['Longitude'].mean()

south_west_corner = Location(latitude=21.501795	, longitude=39.244198)
north_east_corner = Location(latitude=21.771004, longitude=39.220411)
VRP = VehicleRoutePlan(name="Kinza",
                 south_west_corner=south_west_corner,
                 north_east_corner=north_east_corner,
                 vehicles=vehicles,
                 visits=visits)
