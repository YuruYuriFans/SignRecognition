import os
import shutil

# Paths
ABLATION_FOLDER = "ablated_models"
TRAINED_FOLDER = "trained_models"
TUNED_FOLDER = "tuned_models"

# Files to keep in trained_models
KEEP_FILES = {
    "best_mobilenetv4_small_basic.pth",
    "best_minivgg_basic.pth",
    "best_lenet_basic.pth",
    "best_mobilenetv2_025_basic.pth"
}

def clean_folder(folder_path, keep_files=None):
    """
    Recursively deletes all files and folders in folder_path,
    except files listed in keep_files (if provided).
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # If it's a directory → delete entire directory
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted folder: {file_path}")

        # If it's a file → delete unless in keep list
        elif os.path.isfile(file_path):
            if keep_files is None or filename not in keep_files:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"Kept file: {file_path}")

def main():
    print("Cleaning ablated_models folder (recursive)...")
    clean_folder(ABLATION_FOLDER)

    print("\nCleaning tuned_models folder (recursive)...")
    clean_folder(TUNED_FOLDER)

    print("\nCleaning trained_models folder (recursive), keeping 4 baseline files...")
    clean_folder(TRAINED_FOLDER, KEEP_FILES)

    print("\nDone!")

if __name__ == "__main__":
    main()
