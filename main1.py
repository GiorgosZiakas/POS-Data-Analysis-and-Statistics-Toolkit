import pandas as pd
from datetime import datetime
import chardet 

# Load the dataset
file_path = '' # Add the path to the CSV file
data = pd.read_csv(file_path, sep=';')
data.head()
print(data.columns)

# Add functions to detect file encoding and parse custom-formatted CSV files
'''def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        detector = chardet.UniversalDetector()
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

file_path = ''  

# Detect the encoding of the file
file_encoding = detect_encoding(file_path)

# Custom parsing function to manually split the columns
def custom_csv_parser(file_path, encoding, delimiter='\t', quotechar='"'):
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            # Remove the quote characters and split by the delimiter
            stripped_line = line.strip(quotechar).strip('\n')
            split_line = stripped_line.split(delimiter)
            data.append(split_line)
    return data

# Parsing the file with the custom function
parsed_data = custom_csv_parser(file_path, file_encoding)

# The first row contains the column names, so we separate it
column_names = parsed_data[0]

# The rest of the data is the actual data
data_rows = parsed_data[1:]


data = pd.DataFrame(data_rows, columns=column_names)
print(data.columns)
print(data.dtypes)'''


# Replace NaN in 'Category' with 'None'
data['Category'] = data['Category'].fillna('None')

# Exclude rows where Category is 'None' and Item is 'TAKEAWAY', and
# rows where Category is 'Desert - Online' and Item is 'Baklavas (2 pieces)'
data = data[~(((data['Category'] == 'None') & (data['Item'] == 'TAKEAWAY')) |
              ((data['Category'] == 'None') & (data['Item'] == 'DELIVEROO')) |
              ((data['Category'] == 'None') & (data['Item'] == 'UBER')))]

# Unique values of payment ids 
unique_payment_ids = data['Payment ID'].nunique()
print(f"Unique payment ids: {unique_payment_ids}")

def classify_delivery_receipts(data):
    """
    Classifies receipts into delivery and non-delivery categories.

    Args:
    data (DataFrame): The original dataset.

    Returns:
    Tuple of two DataFrames: (delivery_data, non_delivery_data)
    """
    # Identifying delivery receipts (DELIVEROO or UBER in 'None' category)
    delivery_receipts_ids = (
        (data['Category'] == 'None') & data['Item'].isin(['DELIVEROO', 'UBER']) |
        data['Item'].str.endswith('.')
    )
    delivery_receipts_ids = data[delivery_receipts_ids]['Payment ID'].unique()

    # Splitting the dataset into delivery and non-delivery datasets
    delivery_data = data[data['Payment ID'].isin(delivery_receipts_ids)]
    non_delivery_data = data[~data['Payment ID'].isin(delivery_receipts_ids)]

    return delivery_data, non_delivery_data

# Applying the function to the original dataset
delivery_data, non_delivery_data = classify_delivery_receipts(data)

# Showing the first few rows of each dataset for verification
delivery_data.head(), non_delivery_data.head()

# Counting the number of receipts in each dataset
num_delivery_receipts = len(delivery_data['Payment ID'].unique())
num_non_delivery_receipts = len(non_delivery_data['Payment ID'].unique())
num_delivery_receipts, num_non_delivery_receipts

def clean_gross_sales(df):
    """
    Cleans the 'Gross Sales' column and converts it to float.

    Args:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with cleaned 'Gross Sales'.
    """
    df = df.copy()
    df['Gross Sales'] = df['Gross Sales'].replace('[£,]', '', regex=True).astype(float)
    #df['Gross Sales'] = pd.to_numeric(data['Gross Sales'].str.replace('¬£', '', regex=False), errors='coerce')
    df['Qty'] = pd.to_numeric(data['Qty'], errors='coerce') 
    print(df.dtypes)
    return df

def identify_sales_with_drinks(df):
    """
    Identifies sales with drinks based on 'Meal Deal' in modifiers, individual drink sales, or 'Drinks' category.

    Args:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with a new column 'With Drinks' that indicates whether a sale includes drinks.
    """
    df = df.copy()
    df['With Drinks'] = df['Modifiers Applied'].str.contains('Meal Deal', case=False, na=False) | (df['Category'] == 'Drinks')  
    # Include 'Coca Cola 500ml' in the sales with drinks
    df['Is Coca Cola 500ml'] = (df['Category'] == 'None') & (df['Item'].str.contains('Coca Cola 500ml', case=False, na=False))
    df['With Drinks'] = df['With Drinks'] | df['Is Coca Cola 500ml']
    
    return df

def classify_as_drink_sale(df):
    """
    Classifies each receipt in the dataset as with or without drinks.

    Args:
    df (DataFrame): The dataset.

    Returns:
    DataFrame: The dataset with a new column 'Receipt Includes Drinks'.
    """
    df = df.copy()
    def classify_row(row):
        if row['Category'] == 'Drinks':
            return True 
        if pd.notna(row['Modifiers Applied']) and 'Meal Deal' in row['Modifiers Applied']:
            return True 
        if row['Category'] == 'None' and 'Coca Cola 500ml' in row['Item']:
            return True
        return False
    df['Receipt Includes Drinks'] = df.apply(classify_row, axis=1)
    return df
    

# Applying the functions for sales with drinks to the non-delivery dataset or to the delivery dataset
delivery_data_cleaned = clean_gross_sales(delivery_data) # similar you can apply the same function to non_delivery_data
delivery_data_with_drinks= identify_sales_with_drinks(delivery_data_cleaned)
final_delivery_data = classify_as_drink_sale(delivery_data_with_drinks)



# Creating a copy of the filtered non-delivery data for transformations
delivery_data_processed = final_delivery_data.copy()

# Aggregating gross sales for each payment ID
payment_id_aggregated = delivery_data_processed.groupby(['Payment ID']).agg({
    'Gross Sales': 'sum',
    'Qty': 'sum',
    'Receipt Includes Drinks': 'max',  # Since each payment ID is classified as with or without drinks
    'Date': 'first'  # Assuming each payment ID corresponds to a single date
}).reset_index()

# Summarizing this data on a daily basis
daily_sales_summary = payment_id_aggregated.groupby(['Date', 'Receipt Includes Drinks']).agg({
    'Gross Sales': 'sum',
    'Payment ID': 'nunique',
    'Qty': 'sum',
}).reset_index()

# Separating metrics for sales with and without drinks
daily_sales_with_drinks = daily_sales_summary[daily_sales_summary['Receipt Includes Drinks'] == True]
daily_sales_without_drinks = daily_sales_summary[daily_sales_summary['Receipt Includes Drinks'] == False]

# Merging data
daily_sales_merged = daily_sales_with_drinks.merge(
    daily_sales_without_drinks, 
    on='Date', 
    suffixes=('_with', '_without'), 
    how='outer'
)

# Calculate total sales and average metrics
daily_sales_merged['Total Gross Sales'] = daily_sales_merged['Gross Sales_with'].fillna(0) + daily_sales_merged['Gross Sales_without'].fillna(0)
daily_sales_merged['Total Receipts'] = daily_sales_merged['Payment ID_with'].fillna(0) + daily_sales_merged['Payment ID_without'].fillna(0)
daily_sales_merged['Total Items Sold'] = daily_sales_merged['Qty_with'].fillna(0) + daily_sales_merged['Qty_without'].fillna(0)
daily_sales_merged['Average Spent'] = daily_sales_merged['Total Gross Sales'] / daily_sales_merged['Total Receipts']  
daily_sales_merged['Average Items per Receipt'] = (daily_sales_merged['Qty_with'].fillna(0) + daily_sales_merged['Qty_without'].fillna(0)) / daily_sales_merged['Total Receipts']
daily_sales_merged['Average Spent with Drinks'] = daily_sales_merged['Gross Sales_with'] / daily_sales_merged['Payment ID_with']
daily_sales_merged['Average Spent w/o Drinks'] = daily_sales_merged['Gross Sales_without'] / daily_sales_merged['Payment ID_without']
daily_sales_merged['Average Items per Receipt with Drinks'] = daily_sales_merged['Qty_with'].fillna(0) / daily_sales_merged['Payment ID_with']
daily_sales_merged['Average Items per Receipt without Drinks'] = daily_sales_merged['Qty_without'].fillna(0) / daily_sales_merged['Payment ID_without']
daily_sales_merged['Total Qty with Drinks'] = daily_sales_merged['Qty_with'].fillna(0).sum()
daily_sales_merged['Total Qty without Drinks'] = daily_sales_merged['Qty_without'].fillna(0).sum()




# Prepare the final table with required columns
final_table_corrected = daily_sales_merged[[
    'Date', 
    'Total Gross Sales', 
    'Total Receipts', 
    'Average Spent', 
    'Average Items per Receipt', 
    'Gross Sales_with', 
    'Payment ID_with', 
    'Average Spent with Drinks',
    'Average Items per Receipt with Drinks',
    'Gross Sales_without', 
    'Payment ID_without',
    'Average Spent w/o Drinks',
    'Average Items per Receipt without Drinks',
    'Total Qty with Drinks',
    'Total Qty without Drinks'
]]

# Rename the columns for clarity
final_table_corrected.columns = [
    'Date', 
    'Total Gross Sales', 
    'Total Receipts', 
    'Av. Spent', 
    'Av. Items/Receipt', 
    'Gross Sales with Drinks', 
    'Receipts with Drinks',
    'Av. Spent with Drinks', 
    'Av. Items/Receipt with Drinks',  
    'Gross Sales w/o Drinks', 
    'Receipts w/o Drinks',
    'Av. Spent w/o Drinks',
    'Av. Items/Receipt w/o Drinks' ,
    'T.Qty with Drinks',
    'T.Qty w/o Drinks' 
]

# Display the head of the final table to verify the results
final_table_corrected.head(), final_table_corrected['Total Receipts'].sum(), final_table_corrected['Total Gross Sales'].sum()
daily_sales_merged['Total Items Sold'].sum()


excel_file_path = '/path/to/yourfile/name.xlsx'
final_table_corrected.to_excel(excel_file_path, index=False)

# Grouping by 'Category' and 'Item' and calculating the sum of 'Gross Sales' and items sold
category_item_aggregation = final_delivery_data.groupby(['Category', 'Item']).agg({
    'Gross Sales': 'sum',
    'Qty': 'sum'
}).reset_index()

# Calculating the total gross sales from this aggregation
total_gross_sales_by_item = category_item_aggregation['Gross Sales'].sum()
total_items_sold_by_item = category_item_aggregation['Qty'].sum()

print(f"Total gross sales by item and category: {total_gross_sales_by_item}")



# Base meal deal names
base_meal_deals = [
    "Meal Deal with Coke",
    "Meal Deal with Diet Coke",
    "Meal Deal with Fanta Lemon",
    "Meal Deal with Fanta Orange",
    "Meal Deal with Sparkling",
    "Meal Deal with Sprite",
    "Meal Deal with Water",
    "Meal Deal with Zero Coke"
]

# Filter dataset for rows where 'Modifiers Applied' contains 'Meal Deal'
meal_deals_data = final_delivery_data[final_delivery_data['Modifiers Applied'].str.contains('Meal Deal', na=False, case=False)]

# Function to map to base meal deal
def map_to_base_meal_deal(name):
    for base_deal in base_meal_deals:
        if base_deal in name:
            return base_deal
    return "Other Meal Deals"  # Return this if it doesn't match any base meal deal


# Apply the mapping function
meal_deals_data['Base Meal Deal'] = meal_deals_data['Modifiers Applied'].apply(map_to_base_meal_deal)

# Group by the base form of meal deals and aggregate 'Qty' and 'Gross Sales'
meal_deals_base_aggregated = meal_deals_data.groupby('Base Meal Deal').agg({
    'Gross Sales': 'sum',
    'Qty': 'sum'
}).reset_index()

# Further analyze the contribution of each specific item to these meal deals
item_contribution = meal_deals_data.groupby(['Base Meal Deal', 'Item']).agg({
    'Gross Sales': 'sum',
    'Qty': 'sum'
}).reset_index()


# Check if the aggregation resulted in a DataFrame
if isinstance(meal_deals_base_aggregated, pd.DataFrame):
    print("meal_deals_base_aggregated is a DataFrame.")
else:
    print("meal_deals_base_aggregated is not a DataFrame.")


excel_file_path = '/path/to/yourfile/name.xlsx'
# Using ExcelWriter to write to multiple sheets
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    category_item_aggregation.to_excel(writer, sheet_name='Category Item Aggregation', index=False)
    meal_deals_base_aggregated.to_excel(writer, sheet_name='Meal Deal Summary', index=False)
    item_contribution.to_excel(writer, sheet_name='Item Contribution', index=False)

