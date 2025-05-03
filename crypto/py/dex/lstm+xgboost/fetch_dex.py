from dune_client.client import DuneClient
import pandas as pd

# replace with your actual API key
API_KEY = "YOUR API KEY"

# initialize the client
client = DuneClient(api_key=API_KEY)

# define query ID as an integer
QUERY_ID = "YOUR QUERY ID"

# fetch data from dune
result = client.get_latest_result(QUERY_ID)

# access the result property
df = pd.DataFrame(result.result.rows)

# ensure 'date' column is in datetime format
df['day'] = pd.to_datetime(df['day'])

# sort by date
df = df.sort_values('day')

# save to csv for later use
df.to_csv("dex_volume_data.csv", index=False)

print("DEX volume data fetched and saved as 'dex_volume_data.csv'.")

