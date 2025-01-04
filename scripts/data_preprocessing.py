import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

    # Standardize column names by making them lowercase and stripping whitespace
    df.columns = df.columns.str.strip().str.lower()

    # Print the standardized column names
    print("Standardized columns in the dataset:", df.columns.tolist())

    # Convert the date column to datetime if it exists
    if 'orderdate' in df.columns:
        df['orderdate'] = pd.to_datetime(df['orderdate'])

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Check if required columns exist
    required_columns = ['orderitemquantity', 'totalitemquantity']
    if not all(column in df.columns for column in required_columns):
        print(f"Error: The dataset must contain the columns: {required_columns}")
        return None, None

    # Extract relevant features for model training
    features = df[required_columns]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler, df
