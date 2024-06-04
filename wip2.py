import sqlite3
import json
import sys
import os

root_dir = 'F:\\' if sys.platform == 'win32' else '/mnt/f/'
try:
    input_file = os.path.normpath(os.path.join(root_dir, 'spaces', 'resources', 'datasets', 'mybible', 'bibles', sys.argv[1]))
except Exception as e:
    print(f"File not found: {e}")
    sys.exit(1)

output_file = input_file.replace('.mybible', '.jsonl')

# Check if the input file exists
if not os.path.exists(input_file):
    print("File not found:", input_file)
    sys.exit(1)

print(f"Loading {input_file}")
# Connect to the SQLite database
conn = sqlite3.connect(input_file)
cursor = conn.cursor()

# Fetch all rows from the Bible table
cursor.execute("SELECT * FROM Bible")
bible_rows = cursor.fetchall()

# Fetch all rows from the Details table
cursor.execute("SELECT * FROM Details")
details_rows = cursor.fetchall()

# Fetch the schema information for the Details table
cursor.execute("PRAGMA table_info(Details)")
details_schema = cursor.fetchall()

# Create a mapping from column names to their indices
column_mapping = {column[1]: index for index, column in enumerate(details_schema)}

# Helper function to safely get a column value by name
def get_column_value(row, column_name, default="unknown"):
    index = column_mapping.get(column_name)
    return row[index] if index is not None and row[index] is not None else default

# Initialize the data dictionary
data = {}

# Fill the data dictionary by iterating over column_mapping
for column_name in column_mapping.keys():
    data[column_name] = get_column_value(details_rows[0], column_name)

# Add any additional columns that might be missing
expected_columns = [
    "Title", "Description", "Abbreviation", "Comments", "Version", "VersionDate",
    "PublishDate", "Publisher", "Author", "Creator", "Source", "EditorialComments",
    "Language", "RightToLeft", "OT", "NT", "Strong", "VerseRules", "Site",
    "Alt_verse_map", "Apochrapha", "Filename", "Format", "Interlineary", "Markup",
    "Markup_languages", "Morphology", "Resource_id", "Resource_level", "Resource_name",
    "Resource_type", "Resource_work", "RTF", "table_id", "Text_type", "Verse_map"
]

# Ensure no duplicates in expected_columns
expected_columns = list(set(expected_columns))

for column_name in expected_columns:
    if column_name not in data:
        data[column_name] = "unknown"

# Now data contains all expected columns with missing ones set to "unknown"
data_list = [data]

abbreviation = data['Abbreviation']
strongs = data['Strong']
verse_rules = data['VerseRules']    
row_number = 0
for row in bible_rows:
    book = bible_rows[row_number][0]
    chapter = bible_rows[row_number][1]
    verse = bible_rows[row_number][2]
    reference =  f"{abbreviation} {book} {chapter}:{verse}"
    scripture = bible_rows[row_number][3]
    data = {
        "Reference": reference,
        "Book": book,
        "Chapter": chapter,
        "Verse": verse,
        "Scripture": scripture,
        "Strongs": strongs,
        "Verse Rules": verse_rules
    }
    row_number += 1
    print(f"Processing {reference}")
    data_list.append(data)

# Write the data to a JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for obj in data_list:
        f.write(json.dumps(obj) + '\n')

# Close the database connection
conn.close()

print(f"Data has been written to {output_file}.jsonl")
