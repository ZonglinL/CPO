import os
import torch
from safetensors.torch import save_file


from huggingface_hub import snapshot_download

def convert_bin_to_safetensors_recursive(root_directory):
    """
    Recursively goes through a root directory and all its subdirectories,
    finds .bin files, converts them to .safetensors, and then deletes
    the original .bin files.

    Args:
        root_directory (str): The path to the root directory to start processing.
    """
    if not os.path.isdir(root_directory):
        print(f"Error: Root directory not found at {root_directory}")
        return

    print(f"Starting recursive conversion in directory: {root_directory}")
    
    total_converted_files = 0
    total_failed_files = 0

    # os.walk yields a 3-tuple: (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(root_directory):
        print(f"\nScanning directory: {dirpath}")
        
        found_bin_in_current_dir = False
        for filename in filenames:
            if filename.endswith(".bin"):
                found_bin_in_current_dir = True
                bin_filepath = os.path.join(dirpath, filename)
                
                # Determine the new .safetensors filename
                safetensors_filename = filename.replace(".bin", ".safetensors")
                safetensors_filepath = os.path.join(dirpath, safetensors_filename)

                print(f"  Processing file: {filename}")
                print(f"    Input: {bin_filepath}")
                print(f"    Output: {safetensors_filepath}")

                try:
                    # Load the state dictionary from the .bin file
                    # map_location='cpu' is generally safe for loading, especially during conversion
                    state_dict = torch.load(bin_filepath, map_location="cpu")
                    print("    .bin file loaded successfully.")

                    # Save the state dictionary to .safetensors format
                    save_file(state_dict, safetensors_filepath)
                    print("    .safetensors file saved successfully.")

                    # Delete the original .bin file
                    os.remove(bin_filepath)
                    print(f"    Original .bin file '{filename}' deleted.")
                    total_converted_files += 1

                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
                    print(f"    Skipping deletion of {filename} due to error.")
                    total_failed_files += 1
                    continue # Continue to the next file even if one fails
        
        if not found_bin_in_current_dir:
            print("  No .bin files found in this directory.")

    print("\n-------------------------------------------")
    print("Recursive conversion process complete.")
    print(f"Total files converted: {total_converted_files}")
    print(f"Total files failed: {total_failed_files}")
    print("-------------------------------------------")

# --- How to use the script ---
if __name__ == "__main__":
    # Specify the root directory where you want to start the search
    # This should be the top-level folder of your model checkpoints
    
    # For your specific case, if 'controlnet' might be a sub-directory containing .bin files,
    # you might want to point this to the parent of 'controlnet'
    # or even the 'checkpoint-5000' directory.
    
    # Example 1: If .bin files are directly in 'controlnet' or its immediate subfolders like 'controlnet/model_a/x.bin'


    subfolder_path = snapshot_download(
        repo_id="limingcv/reward_controlnet",
        local_dir="reward_controlnet_download",
        repo_type="model",
        allow_patterns = ["checkpoints/"]
    )

    target_root_directory = "reward_controlnet_download"

    # Example 2: If the 'controlnet' itself is a subdirectory and you want to process all model components
    # within 'checkpoint-5000'
    # target_root_directory = "/lustre/fs1/home/zo258499/Control-Anything/reward_controlnet_download/checkpoints/ade20k_reward-model-UperNet-R50/checkpoint-5000"


    convert_bin_to_safetensors_recursive(target_root_directory)

    print("\nRemember to verify the contents of the directories after running.")
