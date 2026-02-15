import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_all_the_features(x_train_orig,y):

    # x_train_orig = x_train_orig[:,0:10]
    if isinstance(x_train_orig, np.ndarray):
        x_train_orig = pd.DataFrame(x_train_orig)
        # add name simply x1, x2, x3, ...
        x_train_orig.columns = [f'x{i}' for i in range(1, x_train_orig.shape[1] + 1)]
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    df = x_train_orig.copy()  
    # get only the first 10 columns
    # df = df.iloc[:,0:100]

    df['y'] = y  # Add the target vector to the DataFrame

    # Discretize y into bins using pd.cut to avoid duplicate bin edges
    y_bins = pd.cut(df['y'], bins=10 )  # Use pd.cut to segment y
    df['y_bins'] = y_bins


    # Determine the layout for subplots
    n_cols = 10  # Number of columns in the grid
    n_rows = (len(df.columns) + n_cols - 1) // n_cols  # Calculate number of rows based on the number of columns

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols , figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()  # Flatten the 2D array of axes so we can iterate through them easily
    for idx, column in enumerate(df.columns[:-2]):  # Exclude 'y' and 'y_bins'
        if pd.api.types.is_numeric_dtype(df[column]):
            # Create stacked histogram with hue based on 'y_bins'
            sns.histplot(data=df, x=column, hue='y_bins', multiple='stack', ax=axes[idx], bins=10, palette="viridis", alpha=0.7)
            axes[idx].set_title(f'  {column} ')
            axes[idx].set_xlabel(column)
            axes[idx].get_legend().remove()

            
    # Hide any extra subplots
    for i in range(len(df.columns) - 2, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()