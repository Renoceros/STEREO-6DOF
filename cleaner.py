#cleaner.py
"""
It'll go through all directories that's mentioned, crawl inside them, and find empty folder tree's
Let's say I put in dir = 'image/preprocessed' it'll go through the stuff inside one by one
oh in preprocessed there's preprocessed/BATCH_10 it'll continue to crawl and it found 2 folders
/image/preprocessed/BATCH_10/image_2 there's nothing and in BATCH_10/image_3/ also nothing
it'll delete BATCH_10

if turns out BATCH_11 has stuff inside image_2 and image_3, it wont do anything to it, have it be a function that's called in main

it doesnt have to just be batches it can be any folder inside of the dir that's mentioned that's empty and isnt storing any files

print out all the empty folders and
do a input Y/N to delete or no
if Y
make rand number gen 1-9 if and print it out
if the next input is the same as the number generated, delete.
"""
import os
import random
import shutil

from numpy import append

def is_folder_empty(folder_path):
    """Check if a folder and all its subfolders contain no files."""
    for root, dirs, files in os.walk(folder_path):
        if files:
            return False
    return True

def find_empty_folders(root_dir):
    """Recursively find all completely empty folders in root_dir."""
    empty_folders = []
    filled_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip if this folder contains files
        if filenames:
            continue
        # Check if all subfolders are also empty (no files at all)
        rel_path = os.path.relpath(dirpath)
        if is_folder_empty(rel_path):
            empty_folders.append(rel_path)
        if not is_folder_empty(rel_path):
            filled_folders.append(rel_path)
    return empty_folders, filled_folders

def confirm_and_delete(folders):
    if not folders:
        print("No empty folders found.")
        return
    
    print("\nEmpty folders found:")
    for f in folders:
        print(" -", f)

    confirm = input("\nDo you want to delete these folders? (Y/N): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    challenge = random.randint(1, 9)
    print(f"Safety check: Type the number {challenge} to confirm deletion.")
    try:
        user_input = int(input("Enter number: "))
        if user_input != challenge:
            print("Incorrect number. Deletion aborted.")
            return
    except ValueError:
        print("Invalid input. Deletion aborted.")
        return

    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"Deleted: {folder}")
        except Exception as e:
            print(f"Failed to delete {folder}: {e}")

def main(dir_to_check):
    print(f"Scanning directory: {dir_to_check}")
    empty_folders, filled_folders = find_empty_folders(dir_to_check)
    print("\nFilled folders found:")
    for f in filled_folders:
        print("- ", f)
    confirm_and_delete(empty_folders)

if __name__ == "__main__":
    main("image/preprocessed")# <-- change this if needed
