import multiprocessing
from tqdm import tqdm
from time import sleep


# Function to process each item
def process_item(item):
    # Replace 'do_something' with actual work
    print(item)
    return item


# Prepare your data
items = range(1000)  # Replace with your list of items

# Set up the multiprocessing pool and specify the number of processes
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

# Map the function over the items using the pool
# We use 'tqdm' to create a progress bar
for _ in tqdm(pool.imap_unordered(process_item, items), total=len(items)):
    pass

# Don't forget to close the pool
pool.close()
pool.join()

print(items[0])
