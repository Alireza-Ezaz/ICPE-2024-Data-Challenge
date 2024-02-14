import pandas as pd

# Load the dataset
file_path = 'path_to_your_files'
callGraphDataframe = pd.read_csv(file_path, on_bad_lines='skip')
print(f"shape of dataframe: {callGraphDataframe.shape}")
initial_count = callGraphDataframe.shape[0]
print(f"Initial record count: {initial_count}")
callGraphDataframe.drop(columns=['service', 'rpc_id', 'rpctype', 'uminstanceid','dminstanceid', 'interface'], inplace=True)

# Separate records with non-numeric 'rt'
invalid_rt_df = callGraphDataframe[~pd.to_numeric(callGraphDataframe['rt'], errors='coerce').notna()]

# Find unique 'traceid's in invalid records
unique_invalid_traceids = set(invalid_rt_df['traceid'])

# Remove all records (both valid and invalid) with these 'traceid's
callGraphDataframe = callGraphDataframe[~callGraphDataframe['traceid'].isin(unique_invalid_traceids)]
print(f"Records removed due to invalid 'rt': {initial_count - callGraphDataframe.shape[0]}")
x = callGraphDataframe.shape[0]

# Further remove records with any empty fields
callGraphDataframe = callGraphDataframe.dropna()
print(f"Records removed due to other empty fields: {x - callGraphDataframe.shape[0]}")
y = callGraphDataframe.shape[0]


print(f"Total records removed: {initial_count - callGraphDataframe.shape[0]}")
print(f"Total records remaining: {callGraphDataframe.shape[0]}")
print(f"percentage of records removed: {(initial_count - callGraphDataframe.shape[0])/initial_count*100}%")

#Save the cleaned dataset
csv_file = 'path_to_save_file'
print("Saving the cleaned dataset...")
callGraphDataframe.to_csv(csv_file, index=False)
print(f"Cleaned dataset saved to {csv_file}")