import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('pred_result.csv')
columns = df.columns

real_df = pd.DataFrame()
pred_df = pd.DataFrame()

for column in columns:
    if 'real' in column:
        real_df[column] = df[column]
    if 'pred' in column:
        pred_df[column] = df[column]

# real_df = real_df.iloc[:100, :]

# def draw_matrix(df, title):
#     aspect_ratio = df.shape[0] / df.shape[1]

#     plt.figure(figsize=(10, 10))
#     plt.imshow(df, cmap='viridis', aspect='auto')
#     plt.colorbar()
#     plt.title(title)
#     plt.xlabel('Area ID')
#     plt.xticks([i for i in range(df.shape[1])])
#     plt.ylabel('Time Step')
#     plt.show()

def draw_matrix_2(df, title):
    plt.figure()
    # plt.axis('equal')
    _, ax = plt.subplots(figsize=(100, 8))
    sns.heatmap(df.T, cmap='YlGnBu', ax=ax, cbar=False)
    ymax = df.shape[0]
    column_num = df.shape[1]
    plt.xlim(0, ymax)
    plt.xticks(np.arange(0, ymax, 50),
                np.arange(0, ymax, 50))
    plt.yticks([i for i in range(0, column_num)], [str(i + 1) for i in range(0, column_num)])

    plt.yticks(fontsize=7)

    im = ax.imshow(df, cmap='viridis', vmin=0, vmax=1)

    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation='horizontal',
                                pad=0.1)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    plt.xlabel('Time Step')
    plt.ylabel('Area ID')
    plt.title(title)
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    # plt.show()
    plt.savefig(title + '.png')

def draw_combined_matrix(real_df, pred_df, title):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(100, 8), sharey=True)
    plt.subplots_adjust(hspace=0.5, wspace=0.5) 
    # First subplot for real_df
    c1 = sns.heatmap(real_df.T, cmap='viridis', ax=axs[0], cbar=False, vmin=0, vmax=19)
    axs[0].set_title('Real Task Flow Data')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Area ID')
    axs[0].set_xticks(np.arange(0, real_df.shape[0], 50))
    axs[0].set_xticklabels(np.arange(0, real_df.shape[0], 50))
    axs[0].set_yticks(np.arange(0, real_df.shape[1]))
    axs[0].set_yticklabels(np.arange(1, real_df.shape[1]+1))
    axs[0].tick_params(axis='y', labelsize=7)

    # Second subplot for pred_df
    c2 = sns.heatmap(pred_df.T, cmap='viridis', ax=axs[1], cbar=False, vmin=0, vmax=19)
    axs[1].set_title('Predicted Task Flow Data')
    axs[1].set_xlabel('Time Step')
    axs[0].set_ylabel('Area ID')
    axs[1].set_xticks(np.arange(0, pred_df.shape[0], 50))
    axs[1].set_xticklabels(np.arange(0, pred_df.shape[0], 50))
    axs[1].set_yticks(np.arange(0, pred_df.shape[1]))
    axs[1].set_yticklabels(np.arange(1, pred_df.shape[1]+1))
    axs[1].tick_params(axis='y', labelsize=7)  # no need to set yticks or yticklabels, shared with axs[0]

    # Common colorbar
    plt.colorbar(axs[1].collections[0], ax=axs, orientation='horizontal', pad=0.1, fraction=0.02)
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    plt.suptitle(title)
    plt.savefig(title + '.png')
    # plt.show()

# draw_matrix(real_df, 'Real Task Flow Data')
# draw_matrix(pred_df, 'Predicted Task FlowData')
if __name__ == '__main__':
    # draw_matrix_2(real_df, 'Real Task Flow Data')
    # draw_matrix_2(pred_df, 'Predicted Task FlowData')
    draw_combined_matrix(real_df, pred_df, 'Real vs Predicted Task Flow Data')

