import os

def rename_songs():
    folder_path = "audio/Songs"
    
    # Ensure we are in the correct directory if relative path is used
    # or use absolute path
    abs_folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(abs_folder_path):
        print(f"Error: Directory {abs_folder_path} does not exist.")
        return

    # Get all mp3 files
    files = [f for f in os.listdir(abs_folder_path) if f.endswith('.mp3')]
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    print(f"Found {len(files)} mp3 files in {abs_folder_path}")
    
    for index, filename in enumerate(files, start=1):
        # Create the new name: song_n_1.mp3, song_n_2.mp3, etc.
        new_name = f"song_n_{index}.mp3"
        
        old_file = os.path.join(abs_folder_path, filename)
        new_file = os.path.join(abs_folder_path, new_name)
        
        try:
            os.rename(old_file, new_file)
            print(f"Renamed: '{filename}' -> '{new_name}'")
        except Exception as e:
            print(f"Error renaming '{filename}': {e}")

if __name__ == "__main__":
    rename_songs()
