import pandas as pd
import requests
import os

def fetch_and_update_data(url, save_path):
    """
    Fetches data from a URL and saves it to the specified path.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Data updated and saved to {save_path}")

if __name__ == "__main__":
    # Example URL to fetch data from (replace with actual data source)
    data_url = "https://example.com/path/to/your/data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_file_path = os.path.join(data_dir, "updated_data.csv")

    fetch_and_update_data(data_url, save_file_path)
