import matplotlib
#matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 8))
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 2,
                             'xtick.labelsize': 18.0,
                             'ytick.labelsize': 18.0,
                             'legend.fontsize': 19.0,
                             'axes.labelsize': 19.0,
                             'axes.titlesize': 19.0,
                             })

# set color
sns.set_palette([(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
                 ])
# print(sns.color_palette("colorblind"))  # dark blue from this palette (first color)  #176D9C
# print(sns.color_palette("tab10"))  # light blue from this palette (last color) #2EABB8
