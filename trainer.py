import sqlite3
import json
import sys
import os
import chardet

model_path = os.path.normpath(os.path.join('..', '..','..','..', 'spaces', 'resources', 'models', 'ai-forever', 'mGPT'))
data_path = os.path.abspath(os.path.join( '..', '..','..','..', 'spaces', 'resources', 'datasets', 'mybible', 'bibles'))

bible_file = os.path.join(data_path, "kjv.bbl.mybible")
if not os.path.exists(bible_file):
    print("File not found:", bible_file)
    sys.exit(1)

# Connect to the SQLite database
conn = sqlite3.connect(bible_file)
cursor = conn.cursor()

# # Detect the file encoding
# with open(bible_file, 'rb') as f:
#     raw_data = f.read()
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']

# Execute the SQL queries from the file
sql_file = bible_file
try:
    with open(sql_file, 'rb') as f:
        cursor.executescript(f.read())
except UnicodeDecodeError as e:
    print(f"Failed to decode file with detected encoding {encoding}: {e}")
    sys.exit(1)

cursor.execute("SELECT * FROM Bible")
bible_rows = cursor.fetchall()
cursor.execute("SELECT * FROM Details")
details_rows = cursor.fetchall()
# # Fetch all rows from both tables
# cursor.execute("SELECT * FROM Bible")
# bible_rows = cursor.fetchall()
# cursor.execute("SELECT * FROM Details")
# details_rows = cursor.fetchall()

# Create a dictionary to store the abbreviations
abbreviation = {row[0]: row[1] for row in details_rows}
Strongs = {row[0]: row[1] for row in details_rows}

# Create a list to store the JSON objects
data_list = []

for row in details_rows:
    data = {
        "Title": row[0],
        "Description": row[1],
        "Abbreviation": row[2],
        "Comments": row[3],
        "Version": row[4],
        "VersionDate": row[5],
        "PublishDate": row[6],
        "Publisher": row[7],
        "Author": row[8],
        "Creator": row[9],
        "Source": row[10],
        "EditorialComments": row[11],
        "Language": row[12],
        "RightToLeft": row[13],
        "OT": row[14],
        "NT": row[15],
        "Strong": row[16],
        "VerseRules": row[17],
        "Site": row[18]
    }
    data_list.append(data)
    
for row in bible_rows:
    data = {
        "Book": row[0],
        "Chapter": row[1],
        "Verse": row[2],
        "Scripture": row[3],
        "Reference": f"{abbreviation} {row[0]} {row[1]}:{row[2]}"
    }
    data_list.append(data)
    
config = {
    "chat_template": "tokenizer",
    "mixed_precision": "fp16",
    "optimizer": "adamw_torch",
    "peft": "true",
    "scheduler": "linear",
    "batch_size": "2",
    "block_size": "1024",
    "epochs": "3",
    "gradient_accumulation": "4",
    "lr": "0.00003",
    "model_max_length": "2048",
    "target_modules": "all-linear"
}

# Create an instance of AutoTrain
# autotrain = AutoTrain(model_path=model_path, data=data_list, config=config)    
# autotrain.train(data_list)

# Save the data to a JSONL file
output_file = sys.argv[1].replace('.mybible', '.jsonl')
with open(output_file, 'w', encoding='utf-8') as f:
    for obj in data_list:
        f.write(json.dumps(obj) + '\n')

# Close the database connection
conn.close()