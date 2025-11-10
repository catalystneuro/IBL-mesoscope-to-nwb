import tarfile
import os
import time
from pathlib import Path


def uncompress_tar(tar_path, extract_to=None):
    """
    Uncompress a .tar or .tar.gz file

    Args:
        tar_path: Path to the tar file
        extract_to: Directory to extract to (default: same directory as tar file)
    """
    try:
        # Start timing
        start_time = time.time()

        # If no extraction path specified, use the tar file's directory
        if extract_to is None:
            extract_to = os.path.dirname(tar_path) or "."

        # Create extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Open and extract the tar file
        with tarfile.open(tar_path, "r:*") as tar:
            print(f"Extracting {tar_path}...")
            tar.extractall(path=extract_to)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            print(f"Successfully extracted to {extract_to}")
            print(f"Extraction completed in {elapsed_time:.2f} seconds")

            # List extracted files
            members = tar.getmembers()
            print(f"\nExtracted {len(members)} items:")
            for member in members[:10]:  # Show first 10 items
                print(f"  - {member.name}")
            if len(members) > 10:
                print(f"  ... and {len(members) - 10} more")

    except tarfile.TarError as e:
        print(f"Error: Failed to extract tar file - {e}")
    except FileNotFoundError:
        print(f"Error: File '{tar_path}' not found")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Uncompress to same directory as tar file
    tar_path = Path(
        r"E:\IBL-data-share\cortexlab\Subjects\SP061\2025-01-27\001\raw_imaging_data_01\imaging.frames.tar.bz2"
    )
    output_dir = tar_path.parent
    uncompress_tar(tar_path)

    # Uncompress to specific directory
    # uncompress_tar("archive.tar.gz", extract_to="./extracted_files")
