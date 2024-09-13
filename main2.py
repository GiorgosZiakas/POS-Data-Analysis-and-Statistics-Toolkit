import pandas as pd
from datetime import datetime
import chardet

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self, sep=';'):
        """Loads the dataset from a CSV file."""
        self.data = pd.read_csv(self.file_path, sep=sep)
        print(self.data.columns)

    def clean_data(self):
        """Cleans data by replacing NaN in 'Category' and filtering rows."""
        self.data['Category'] = self.data['Category'].fillna('None')
        self.data = self.data[~((self.data['Category'] == 'Desert - Online') & (self.data['Item'] == 'Baklavas (2 pieces)'))]
        self.data['Includes_TAKEAWAY'] = self.data.groupby('Payment ID')['Item'].transform(lambda x: 'TAKEAWAY' in x.values)

    def exclude_takeaway(self):
        """Excludes 'TAKEAWAY' sales."""
        self.data = self.data[self.data['Item'] != 'TAKEAWAY']
        return self.data


class ReceiptClassifier:
    def __init__(self, data):
        self.data = data

    def classify_receipts(self):
        """Classifies receipts into delivery and non-delivery categories."""
        delivery_receipts_ids = ((self.data['Category'] == 'None') &
                                 self.data['Item'].isin(['DELIVEROO', 'UBER']) |
                                 self.data['Item'].str.endswith('.'))
        delivery_receipts_ids = self.data[delivery_receipts_ids]['Payment ID'].unique()
        delivery_data = self.data[self.data['Payment ID'].isin(delivery_receipts_ids)]
        non_delivery_data = self.data[~self.data['Payment ID'].isin(delivery_receipts_ids)]
        return delivery_data, non_delivery_data


class SalesAnalyzer:
    def __init__(self, data):
        self.data = data

    def clean_gross_sales(self):
        """Cleans 'Gross Sales' column by removing unwanted characters and converting to float."""
        self.data['Gross Sales'] = self.data['Gross Sales'].replace('[£,]', '', regex=True).astype(float)
        self.data['Qty'] = pd.to_numeric(self.data['Qty'], errors='coerce')

    def identify_sales_with_drinks(self):
        """Identifies sales with drinks based on certain conditions."""
        self.data['With Drinks'] = self.data['Modifiers Applied'].str.contains('Meal Deal', case=False, na=False) | (
                self.data['Category'] == 'Drinks')
        self.data['Is Coca Cola 500ml'] = (self.data['Category'] == 'None') & \
                                          (self.data['Item'].str.contains('Coca Cola 500ml', case=False, na=False))
        self.data['With Drinks'] = self.data['With Drinks'] | self.data['Is Coca Cola 500ml']

    def classify_as_drink_sale(self):
        """Classifies each receipt as with or without drinks."""
        def classify_row(row):
            if row['Category'] == 'Drinks':
                return True
            if pd.notna(row['Modifiers Applied']) and 'Meal Deal' in row['Modifiers Applied']:
                return True
            if row['Category'] == 'None' and 'Coca Cola 500ml' in row['Item']:
                return True
            return False

        self.data['Receipt Includes Drinks'] = self.data.apply(classify_row, axis=1)

    def summarize_daily_sales(self):
        """Aggregates and summarizes sales data on a daily basis."""
        payment_id_aggregated = self.data.groupby(['Payment ID']).agg({
            'Gross Sales': 'sum',
            'Qty': 'sum',
            'Receipt Includes Drinks': 'max',
            'Date': 'first'
        }).reset_index()

        daily_sales_summary = payment_id_aggregated.groupby(['Date', 'Receipt Includes Drinks']).agg({
            'Gross Sales': 'sum',
            'Payment ID': 'nunique',
            'Qty': 'sum',
        }).reset_index()

        return daily_sales_summary


class TakeawayAnalysis:
    def __init__(self, data):
        self.data = data

    def analyze_takeaway(self):
        """Analyzes takeaway sales with and without drinks."""
        with_drinks = self.data[self.data['Receipt Includes Drinks']]
        without_drinks = self.data[~self.data['Receipt Includes Drinks']]

        takeaway_with_drinks_items = with_drinks[with_drinks['Includes_TAKEAWAY']]
        takeaway_without_drinks_items = without_drinks[without_drinks['Includes_TAKEAWAY']]

        takeaway_with_drinks_summary = takeaway_with_drinks_items.groupby('Item').agg({
            'Qty': 'sum',
            'Gross Sales': 'sum'
        }).reset_index()

        takeaway_without_drinks_summary = takeaway_without_drinks_items.groupby('Item').agg({
            'Qty': 'sum',
            'Gross Sales': 'sum'
        }).reset_index()

        return takeaway_with_drinks_summary, takeaway_without_drinks_summary


class MealDealAnalyzer:
    def __init__(self, data):
        self.data = data
        self.base_meal_deals = [
            "Meal Deal with Coke", "Meal Deal with Diet Coke", "Meal Deal with Fanta Lemon",
            "Meal Deal with Fanta Orange", "Meal Deal with Sparkling", "Meal Deal with Sprite",
            "Meal Deal with Water", "Meal Deal with Zero Coke"
        ]

    def analyze_meal_deals(self):
        """Analyzes meal deals by mapping to base deals and aggregating sales."""
        meal_deals_data = self.data[self.data['Modifiers Applied'].str.contains('Meal Deal', na=False, case=False)]

        meal_deals_data['Base Meal Deal'] = meal_deals_data['Modifiers Applied'].apply(self.map_to_base_meal_deal)
        meal_deals_aggregated = meal_deals_data.groupby('Base Meal Deal').agg({
            'Gross Sales': 'sum',
            'Qty': 'sum'
        }).reset_index()

        item_contribution = meal_deals_data.groupby(['Base Meal Deal', 'Item']).agg({
            'Gross Sales': 'sum',
            'Qty': 'sum'
        }).reset_index()

        return meal_deals_aggregated, item_contribution

    def map_to_base_meal_deal(self, name):
        for base_deal in self.base_meal_deals:
            if base_deal in name:
                return base_deal
        return "Other Meal Deals"


class ExcelExporter:
    @staticmethod
    def export_to_excel(dfs, sheet_names, excel_file_path):
        """Exports multiple DataFrames to an Excel file with specified sheet names."""
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            for df, sheet_name in zip(dfs, sheet_names):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Data exported successfully to {excel_file_path}")


# Example usage
file_path = '/path/to/yourfile.csv'

# Step 1: Load and clean data
processor = DataProcessor(file_path)
processor.load_data()
processor.clean_data()
data_without_takeaway = processor.exclude_takeaway()

# Step 2: Classify receipts into delivery and non-delivery
classifier = ReceiptClassifier(data_without_takeaway)
delivery_data, non_delivery_data = classifier.classify_receipts()

# Step 3: Analyze non-delivery sales
analyzer = SalesAnalyzer(non_delivery_data)
analyzer.clean_gross_sales()
analyzer.identify_sales_with_drinks()
analyzer.classify_as_drink_sale()
daily_sales_summary = analyzer.summarize_daily_sales()

# Step 4: Analyze takeaway sales
takeaway_analysis = TakeawayAnalysis(non_delivery_data)
takeaway_with_drinks_summary, takeaway_without_drinks_summary = takeaway_analysis.analyze_takeaway()

# Step 5: Analyze meal deals
meal_deal_analyzer = MealDealAnalyzer(non_delivery_data)
meal_deals_aggregated, item_contribution = meal_deal_analyzer.analyze_meal_deals()

# Step 6: Export results to Excel
excel_file_path = '/path/to/yourfile/name.xlsx'
dfs = [takeaway_with_drinks_summary, takeaway_without_drinks_summary, meal_deals_aggregated, item_contribution]
sheet_names = ['TAKEAWAY With Drinks', 'TAKEAWAY Without Drinks', 'Meal Deal Summary', 'Item Contribution']
ExcelExporter.export_to_excel(dfs, sheet_names, excel_file_path)

    

    
    
    




 
