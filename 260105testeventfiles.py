#!/usr/bin/env python
import glob
import pandas as pd
import os

EVENT_DIR = "out/251110/event_datasets_benchmark"

print("=" * 70)
print("CHECKING EVENT FILES")
print("=" * 70)

# 1. Check if directory exists
if not os.path.exists(EVENT_DIR):
    print(f"ERROR: Directory not found: {EVENT_DIR}")
    print("Run preprocess_event_level.py first!")
    exit(1)

# 2. Count files
event_files = glob.glob(f"{EVENT_DIR}/*_events.csv")
print(f"Found {len(event_files)} *_events.csv files")
print()

if len(event_files) == 0:
    print("No event files found!")
    print("Files in directory:")
    for f in os.listdir(EVENT_DIR):
        print(f"  - {f}")
    exit(1)

# 3. Check first 3 files
print("Checking first 3 files:")
print()

for i, file_path in enumerate(event_files[:3]):
    print(f"{i+1}. {os.path.basename(file_path)}")
    
    try:
        # Quick read
        df = pd.read_csv(file_path, nrows=3)
        
        # Basic info
        print(f"   Rows: {len(pd.read_csv(file_path)):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Column names: {df.columns.tolist()}")
        
        # Check for required columns
        required = ['case_id', 'activity', 'timestamp']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"   ⚠ MISSING REQUIRED: {missing}")
        
        # Check for unwanted prefix columns
        unwanted = ['prefix_index', 'remaining_time', 'elapsed_time']
        found_unwanted = [col for col in unwanted if col in df.columns]
        if found_unwanted:
            print(f"   ⚠ HAS PREFIX COLUMNS: {found_unwanted}")
        
        # Sample data
        print(f"   Sample data:")
        print(df.head(2).to_string())
        
    except Exception as e:
        print(f"   ERROR reading: {e}")
    
    print()

print("=" * 70)
if len(event_files) > 3:
    print(f"... and {len(event_files)-3} more files")
print("If everything looks good, update dataset_confs.py and run benchmark!")