"""
split_for_github.py
===================
Splits a large file (like models.zip) into 1.5 GB chunks so they 
can be uploaded to GitHub Releases without hitting the 2GB limit.
"""

import os
import sys

# 1.5 GB in bytes
CHUNK_SIZE = int(1.5 * 1024 * 1024 * 1024)

def split_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    file_size = os.path.getsize(filepath)
    if file_size <= CHUNK_SIZE:
        print(f"File is {file_size / (1024**3):.2f} GB. It is already under the 2GB limit!")
        return

    print(f"Splitting {filepath} ({file_size / (1024**3):.2f} GB) into 1.5 GB chunks...")
    
    part_num = 1
    with open(filepath, 'rb') as f_in:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
                
            out_filename = f"{filepath}.part{part_num}"
            print(f"  Writing {out_filename}...")
            
            with open(out_filename, 'wb') as f_out:
                f_out.write(chunk)
                
            part_num += 1

    print("=" * 60)
    print(f"Done! Split into {part_num - 1} parts.")
    print("Upload all these .part files to your GitHub Release.")
    print("=" * 60)

if __name__ == "__main__":
    target = "models.zip"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    
    split_file(target)
