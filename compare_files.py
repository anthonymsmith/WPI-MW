import pandas as pd
import sys

def compare_csv_files(file1, file2):
    """
    Compares two CSV files and identifies differences in content.
    """
    try:
        # Load the CSV files into Pandas DataFrames
        df1 = pd.read_csv(file1)
        print('file1 loaded')
        df2 = pd.read_csv(file2)
        print('file2 loaded')


    # Compare row counts
        if df1.shape != df2.shape:
            print(f"File dimensions differ: {file1} has {df1.shape}, {file2} has {df2.shape}")

        # Find differences
        differences = pd.concat([df1, df2]).drop_duplicates(keep=False)

        if not differences.empty:
            print("\nDifferences found between the two CSV files:")
            print(differences)
        else:
            print("The two CSV files are identical.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except pd.errors.EmptyDataError:
        print("Error: One or both of the files are empty.")
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_csv_files.py <file1> <file2>")
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        compare_csv_files(file1, file2)