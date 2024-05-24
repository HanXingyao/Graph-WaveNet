import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


maps = ['BB', 'JA', 'TG']

result = {
    'JA': [86.8410875, 82.1998625, 73.16515],
    'TG': [85.85431667, 75.05631667, 69.989],
    'BB': [83.81741667, 72.33076667, 56.21876667]
}
labels = list(result.keys())
window_5 = [result[key][0] for key in labels]
window_10 = [result[key][1] for key in labels]
window_15 = [result[key][2] for key in labels]

x = np.arange(len(labels))
width = 0.6

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/3, window_5, width/3, label='Window len 5')
rects2 = ax.bar(x, window_10, width/3, label='Window len 10')
rects3 = ax.bar(x + width/3, window_15, width/3, label='Window len 15')

ax.set_ylabel('Precision')
ax.set_title('Precision by map and window size')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 100)

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height,
                '%d' % int(height), ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

fig.tight_layout()

plt.show()
