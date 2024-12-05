import os
import shutil
import subprocess

def delete_file(file_path):
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File not found (skipped): {file_path}")

def delete_folder(folder_path):
    """Delete a folder and its contents if it exists."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder not found (skipped): {folder_path}")

def clear_pycache(directory):
    """Delete all __pycache__ directories recursively."""
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"Deleted __pycache__: {pycache_path}")

def reset_environment(env_path):
    """Recreate the virtual environment."""
    print("Recreating virtual environment...")
    if os.path.exists(env_path):
        shutil.rmtree(env_path)
        print(f"Deleted virtual environment: {env_path}")
    subprocess.run(["python", "-m", "venv", env_path])
    print(f"Recreated virtual environment at: {env_path}")
    subprocess.run([os.path.join(env_path, "Scripts", "pip"), "install", "-r", "requirements.txt"])
    print("Reinstalled dependencies.")

def reset_project():
    # Paths to reset
    model_path = "models/custom/custom_model.pt"
    loss_log_path = "QuickAid_loss_log.xlsx"
    loss_chart_path = "QuickAid_loss_chart.png"
    duplicates_dir = "duplicates"
    data_dirs = [
        "data/images/train",
        "data/images/test",
        "data/images/valid"
    ]
    env_dir = "env"  # Virtual environment folder

    # Delete model weights
    delete_file(model_path)

    # Delete old training logs and charts
    delete_file(loss_log_path)
    delete_file(loss_chart_path)

    # Delete duplicates folder
    delete_folder(duplicates_dir)

    # Clear dataset folders
    for data_dir in data_dirs:
        delete_folder(data_dir)

    # Clear __pycache__ directories
    clear_pycache(".")

    # Recreate virtual environment
    reset_environment(env_dir)

    print("\nProject reset complete! You can now add your new dataset and start fresh.")

if __name__ == "__main__":
    reset_project()
