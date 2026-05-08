import matplotlib.pyplot as plt
import pandas as pd

def plot_value_from_dataframe(df, columns, key_column, fig_size=(10,10)):
    """create a plot from a dataframe, that has timepoints as columns 
       and each row represents a value at a certain timepoint the plot 
       will be returned as a figure object"""
    fig, ax = plt.subplots(figsize=fig_size)
    for _, row in df.iterrows():
        y = [row[col] for col in columns]
        x = [col for col in columns]
        ax.plot(x, y, label=row[key_column])
    return fig