import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter, LogLocator, ScalarFormatter, FuncFormatter

# Use SciencePlots styles
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])


def load_data_from_dir(dir_path):
    # Check if the directory exists
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return None
    else:
        
        data_lst = []
        
        for (root, dirs, files) in os.walk(dir_path):
            for file in files:                
                if file.endswith(".csv"): 
                    path = os.path.join(dir_path, file)
                    
                    try:
                        # Read the CSV. Assuming the first column is the index and the second is the data.
                        # The 'header=0' tells pandas the first row contains column names.
                        df = pd.read_csv(path, index_col=0)
                        
                        # Append to a list
                        data_lst.append(df)
                    
                    except Exception as e:
                        print(f" - Error reading or processing {filename}: {e}")
                        
    return data_lst

def data_to_numpy(df_lst):
    mat_lst = []
    for df in df_lst:
        mat_lst.append(df.to_numpy())
        
    return np.vstack(mat_lst)


def main():
    
    path_dict = {}
    path_dict['bgbg'] = "timings/bgbg"
    path_dict['multi'] = "timings/multi_robot"
    path_dict['multi_tamp'] = "timings/multi_robot_tamp"
    
    
    time_dict = {}    
    # Load df data
    for key, path in path_dict.items():
        df_lst = load_data_from_dir(path)
        time_dict[key] = data_to_numpy(df_lst)
        
        print (f' Statistics for {key}: {time_dict[key].mean()}, and {time_dict[key].std()} ')
    
    data = [time_dict['bgbg'].squeeze(), 
            time_dict['multi'].squeeze(), 
            time_dict['multi_tamp'].squeeze()]
    
    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    # Small dot outliers
    flierprops = dict(marker='.', color='black', markersize=2, linestyle='none')

    bp = ax.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor="white"),
        medianprops=dict(color="black", linewidth=1),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=flierprops
    )

    # X-axis labels
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['BGBG \n(Ours)', 'MRPP \nPick & Place', 'MRPP \nTAMP'])
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    ax.set_ylabel("Duration [s]")

    # --- Log scale ---
    ax.set_yscale('log')

    # Minor ticks for log y-axis
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=10))
    # ax.tick_params(axis='y', which='minor', length=3, right=False)

    # Major tick formatting
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

    # --- Full-width gridlines ---
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.8, color='gray')  # major horizontal
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, color='gray')  # minor horizontal

    plt.tight_layout()
    plt.show()
    return 0



if __name__ == "__main__":
    main()                    
