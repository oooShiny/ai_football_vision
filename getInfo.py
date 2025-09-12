import sys
import time
import ollama
import os
import cv2
import numpy as np
import pathlib
from pathlib import Path
from typing import List
from typing import Optional, Tuple

#llm = 'gpt-oss'
#llm = 'llama3.2-vision'
#llm = 'llava'
llm = 'qwen2.5vl'
#llm = 'minicpm-v'
#llm = 'gemma3:12b'

# -----------------------------------------------------------------------------
def parseImg(videoPath: str) -> str:
    with open(videoPath, 'rb') as f:
        image_bytes = f.read()

    # Send the image and prompt to the VLM
    response = client.chat(
            #model='llama3.2-vision', # Or 'llava'
            model=llm, 
            messages=[
#                {'role': 'user', 'content': 'What is the text in this image?', 'images': [image_bytes]}
                {'role': 'user', 'content': 'Find all the text in this image and print out in a single line of text', 'images': [image_bytes]}
            ],
    )

    f.close
    
    # Extract the extracted text
    extracted_text = response['message']['content']
    return extracted_text

# -----------------------------------------------------------------------------

def get_files_by_type_recursive(directory_path: str, file_type: str) -> List[str]:
    """
    Get a list of files in a directory and its subdirectories that match the specified file type.
    
    Args:
        directory_path (str): Path to the directory to search recursively
        file_type (str): File extension to filter by (e.g., '.txt', '.py', '.pdf')
                        Can be provided with or without the leading dot
    
    Returns:
        List[str]: List of file paths that match the specified file type
    """
    # Convert to Path object
    dir_path = Path(directory_path)
    
    # Check if directory exists
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"'{directory_path}' is not a directory")
    
    # Ensure file_type starts with a dot
    if not file_type.startswith('.'):
        file_type = '.' + file_type
    
    # Use glob to find all matching files recursively
    pattern = f"**/*{file_type}"
    matching_files = [str(p) for p in dir_path.glob(pattern) if p.is_file()]
    
    return sorted(matching_files)

# -----------------------------------------------------------------------------

def extract_first_frame(video_path: str, output_path: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Extract the first frame from an MP4 video file and save it as PNG.
    
    Args:
        video_path (str): Path to the input MP4 video file
        output_path (str, optional): Path where to save the PNG frame. 
                                   If None, uses video filename with .png extension
    
    Returns:
        Tuple[bool, Optional[np.ndarray]]: 
            - Success status (True if successful, False otherwise)
            - The frame as numpy array (None if failed)
    """
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Error: Video file '{video_path}' not found")
        return False, None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False, None
    
    try:
        # Read the first frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read the first frame from the video")
            return False, None
        
        # Generate output path if not provided
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = video_path_obj.with_suffix('.png')
        
        # Save frame as PNG
        success = cv2.imwrite(str(output_path), frame)
        
        if success:
            #print(f"First frame successfully saved to: {output_path}")
            return True, frame
        else:
            print(f"Error: Failed to save frame to {output_path}")
            return False, None
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return False, None
    
    finally:
        # Always release the video capture object
        cap.release()


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ---------------------------------- main -------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Initialize the Ollama client
    client = ollama.Client(host='http://localhost:11434') # Adjust host if needed

    try:
        # get a list of all .mp4 files recursively
        text_files = get_files_by_type_recursive(".", "mp4")
        print("\nmp4 files in current directory and subdirectories:")
        for file in text_files:
            print(f"  {file}")
            
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")


    # loop for each video file
    
    for video_file in text_files:

        fpath = str(pathlib.Path(video_file).parent)
        fname = str(pathlib.Path(video_file).stem)
        ftype = str(pathlib.Path(video_file).suffix)
        
        imgFile = fpath + "/" + fname + ".png"
        txtFile = fpath + "/" + fname + ".txt"
    
        # Extract and save first frame
        #print("1. Extracting first frame and saving as PNG:")
        success, frame = extract_first_frame(video_file, imgFile)
    
        #if success:
        #    print(f"    Extracted {imgFile} from {video_file}")

        start_time = time.perf_counter() # Or time.time() for less precise measurements

        print(f"Processing {video_file} ... ", end='')
        sys.stdout.flush()
        
        # Extract the text from img using LLM            
        extracted_text = parseImg(imgFile)

        end_time = time.perf_counter() # Or time.time()
        elapsed_time = end_time - start_time
        
        print(f" ({elapsed_time:.1f}s) ")
        print(f"\"{extracted_text}\"")
        
        with open(txtFile, 'w') as f1:
            f1.write(extracted_text)

        f1.close

