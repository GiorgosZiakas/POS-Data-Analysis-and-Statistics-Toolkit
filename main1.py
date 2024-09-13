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

    def detect_encoding(self):
        """Detects file encoding."""
        with open(self.file_path, 'rb') as file:
            detector = chardet.UniversalDetector()
            for line in file:
                detector.feed(line)
                if detector.done:
                    break
            detector.close()
        return detector.result['encoding']

    def clean_data(self):
        """Cleans data by replacing NaN in 'Category' and filtering rows."""
        self.data['Category'] = self.data['Category'].fillna('None')
        self.data = self.data[~(((self.data['Category'] == 'None') & (self.data['Item'] == 'TAKEAWAY')) |
                                ((self.data['Category'] == 'None') & (self.data['Item'] == 'DELIVEROO')) |
                                ((self.data['Category'] == 'None') & (self.data['Item'] == 'UBER')))]
        print(f"Unique payment ids: {self.data['Payment ID'].nunique()}")

    def clean_gross_sales(self):
        """Cleans 'Gross Sales' column by removing unwanted characters and converting to float."""
        self.data['Gross Sales'] = self.data['Gross Sales'].replace('[£,]', '', regex=True).astype(float)
        self.data['Qty'] = pd.to_numeric(self.data['Qty'], errors='coerce')
        print(self.data.dtypes)


class ReceiptClassifier:
    def __init__(self, data):
        self.data = data
        self.delivery_data = None
        self.non_delivery_data = None

    def classify_receipts(self):
        """Classifies receipts into delivery and non-delivery categories."""
        delivery_receipts_ids = ((self.data['Category'] == 'None') &
                                 self.data['Item'].isin(['DELIVEROO', 'UBER']) |
                                 self.data['Item'].str.endswith('.'))
        delivery_receipts_ids = self.data[delivery_receipts_ids]['Payment ID'].unique()
        self.delivery_data = self.data[self.data['Payment ID'].isin(delivery_receipts_ids)]
        self.non_delivery_data = self.data[~self.data['Payment ID'].isin(delivery_receipts_ids)]
        return self.delivery_data, self.non_delivery_data


class SalesAnalyzer:
    def __init__(self, data):
        self.data = data

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

        daily_sales_with_drinks = daily_sales_summary[daily_sales_summary['Receipt Includes Drinks'] == True]
        daily_sales_without_drinks = daily_sales_summary[daily_sales_summary['Receipt Includes Drinks'] == False]

        # Merging the data
        daily_sales_merged = daily_sales_with_drinks.merge(
            daily_sales_without_drinks,
            on='Date',
            suffixes=('_with', '_without'),
            how='outer'
        )

        # Calculate totals and averages
        daily_sales_merged['Total Gross Sales'] = daily_sales_merged['Gross Sales_with'].fillna(0) + \
                                                  daily_sales_merged['Gross Sales_without'].fillna(0)
        daily_sales_merged['Total Receipts'] = daily_sales_merged['Payment ID_with'].fillna(0) + \
                                               daily_sales_merged['Payment ID_without'].fillna(0)
        daily_sales_merged['Average Spent'] = daily_sales_merged['Total Gross Sales'] / \
                                              daily_sales_merged['Total Receipts']
        daily_sales_merged['Average Items per Receipt'] = (daily_sales_merged['Qty_with'].fillna(0) +
                                                           daily_sales_merged['Qty_without'].fillna(0)) / \
                                                          daily_sales_merged['Total Receipts']
        return daily_sales_merged


class ExcelExporter:
    @staticmethod
    def export_to_excel(dfs, sheet_names, excel_file_path):
        """Exports multiple DataFrames to an Excel file with specified sheet names."""
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            for df, sheet_name in zip(dfs, sheet_names):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Data exported successfully to {excel_file_path}")


# Example usage
file_path = '/path/to/yourfile.csv'

# Step 1: Load and clean data
processor = DataProcessor(file_path)
processor.load_data()
processor.clean_data()
processor.clean_gross_sales()

# Step 2: Classify receipts into delivery and non-delivery
classifier = ReceiptClassifier(processor.data)
delivery_data, non_delivery_data = classifier.classify_receipts()

# Step 3: Analyze sales data
analyzer = SalesAnalyzer(delivery_data)
analyzer.identify_sales_with_drinks()
analyzer.classify_as_drink_sale()
daily_sales_merged = analyzer.summarize_daily_sales()

# Step 4: Export data to Excel
excel_file_path = '/path/to/yourfile/name.xlsx'
dfs = [delivery_data, daily_sales_merged]
sheet_names = ['Delivery Data', 'Daily Sales Summary']
ExcelExporter.export_to_excel(dfs, sheet_names, excel_file_path)

