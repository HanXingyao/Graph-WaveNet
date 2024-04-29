import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('map-wave.csv')
columns = df.columns
print(columns)

real_df = pd.DataFrame()
pred_df = pd.DataFrame()

for column in columns:
    if 'real' in column:
        real_df[column] = df[column]
    if 'pred' in column:
        pred_df[column] = df[column]

fig, ax = plt.subplots(4, 5, figsize=(10, 30))
for i in range(19):
    row = i // 5  
    col = i % 5 
    ax[row, col].plot(real_df.iloc[:, i], label='real')
    ax[row, col].plot(pred_df.iloc[:, i], label='pred')
    if i == 0:
        ax[row, col].legend()
    ax[row, col].set_title(f'Area {i}')
    ax[row, col].set_ylabel('task number per 10 minutes')
    # ax[row, col].set_ylim(0, 6)

ax[-1, -1].axis('off')

plt.subplots_adjust(wspace=0.5)

# plt.show()
plt.savefig('map-pred_result.png')
