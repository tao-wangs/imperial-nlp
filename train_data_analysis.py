import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch 

# csv has format:
# <par_id> <tab> <art_id> <tab> <keyword> <tab> <country_code> <tab> <text> <tab> <label>

# Frequency of class labels 
dataset = pd.read_csv("data/dontpatronizeme_pcl.tsv", 
                       sep="\t", # split by tab character
                       names=["par_id", "keyword", "text", "label"], # relevant data
                       index_col="par_id", # index by par_id
                       usecols=[0, 2, 4, 5], # ignore art_id and country_code
                       skiprows=4) # data starts at line 5 

# {0, 1} -> no PCL 
# {2, 3, 4} -> PCL 
n_pcl_examples = (dataset["label"] > 1.5).sum()
n_non_pcl_examples = (dataset["label"] < 1.5).sum()  

print(f"Number of examples containing PCL: {n_pcl_examples}")
print(f"Number of examples not containing PCL: {n_non_pcl_examples}")

# Frequency of keywords 
unique_keywords = list(np.unique(dataset["keyword"]))
n_unique_keywords = len(unique_keywords)
print(f"Unique Keywords: {unique_keywords}")
print(f"Number of unique keywords: {n_unique_keywords}")

keyword_pcl_dict = dict()
keyword_non_pcl_dict = dict()

keyword_pcl_dict_percentage = dict()
keyword_non_pcl_dict_percentage = dict()

for keyword in unique_keywords:
    n_examples_with_keyword = dataset[dataset["keyword"] == keyword]
    n_examples_with_keyword_pcl = (n_examples_with_keyword["label"] > 1.5).sum()
    n_examples_with_keyword_non_pcl = (n_examples_with_keyword["label"] < 1.5).sum()
    keyword_pcl_dict[keyword] = n_examples_with_keyword_pcl
    keyword_pcl_dict_percentage[keyword] = round((n_examples_with_keyword_pcl / n_pcl_examples) * 100, 1)
    
    keyword_non_pcl_dict[keyword] = n_examples_with_keyword_non_pcl
    keyword_non_pcl_dict_percentage[keyword] = round((n_examples_with_keyword_non_pcl / n_non_pcl_examples) * 100, 1)
    
# Creating the bar chart
plt.bar(unique_keywords, keyword_non_pcl_dict.values(), width=0.35, alpha=0.7, label='No PCL')
plt.bar(unique_keywords, keyword_pcl_dict.values(), width=0.35, alpha=0.7, label='PCL')
plt.xticks(rotation=45)

# Adding title and labels
plt.title('Frequency of Keywords in PCL vs Non PCL')
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('data_analysis/keyword_freq.png')
plt.show()

# Percentages would be better because of the class label imbalance 
x = np.arange(len(unique_keywords)) # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects_pcl = ax.bar(x - width/2, keyword_pcl_dict_percentage.values(), width, label='Positive class (PCL)')
rects_non_pcl = ax.bar(x + width/2, keyword_non_pcl_dict_percentage.values(), width, label='Negative class (No PCL)')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage of total PCL/No PCL examples')
ax.set_title('Class labels by keyword')
ax.set_xticks(x)
ax.set_xticklabels(unique_keywords, ha='center')
ax.legend()

# Function to add labels on the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects_pcl)
autolabel(rects_non_pcl)

fig.tight_layout()
plt.savefig('data_analysis/keyword_perc.png')
plt.show()

# Visualisation between label class and paragraph length
dataset["par_length"] = dataset["text"].apply(lambda x: len(str(x).split()))
positives = dataset[dataset["label"] > 1.5]
negatives = dataset[dataset["label"] < 1.5]

import seaborn as sns

sns.set_theme(style="whitegrid")
sns.histplot(list(positives["par_length"]), bins=np.linspace(0, 200, 100), alpha=0.7, label='PCL')
sns.histplot(list(negatives["par_length"]), bins=np.linspace(0, 200, 100), alpha=0.7, label='No PCL')
plt.xlabel('Value')
plt.ylabel('Normalised Frequency')
plt.title('Distribution of text lengths by class', fontsize=15)
plt.legend()
plt.savefig('data_analysis/text_len_dist.png')
plt.show()

data = pd.DataFrame({
    'Keyword': unique_keywords * 2,
    'Percentage': list(keyword_pcl_dict_percentage.values()) + list(keyword_non_pcl_dict_percentage.values()),
    'Class': ['PCL'] * len(unique_keywords) + ['No PCL'] * len(unique_keywords)
})


barplot = sns.barplot(
    x='Keyword', 
    y='Percentage', 
    hue='Class', 
    data=data, 
    palette='muted', 
    edgecolor='.6'
)

# Add value labels on top of the bars
# for p in barplot.patches:
#     barplot.annotate(format(p.get_height(), '.1f') + '%', 
#                      (p.get_x() + p.get_width() / 2., p.get_height()), 
#                      ha = 'center', va = 'center', 
#                      xytext = (0, 9), 
#                      textcoords = 'offset points')

# Improve the legibility of the plot
plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')
plt.ylabel('Percentage of Total Examples')
plt.xlabel('Keyword')
plt.title('Percentage of Class Labels by Keyword')

# Show the plot
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()