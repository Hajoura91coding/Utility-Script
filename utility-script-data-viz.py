# %% [code] {"jupyter":{"outputs_hidden":false}}
# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:52.515397Z","iopub.execute_input":"2023-11-26T08:52:52.515864Z","iopub.status.idle":"2023-11-26T08:52:52.521874Z","shell.execute_reply.started":"2023-11-26T08:52:52.515828Z","shell.execute_reply":"2023-11-26T08:52:52.520686Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:52.524007Z","iopub.execute_input":"2023-11-26T08:52:52.524359Z","iopub.status.idle":"2023-11-26T08:52:53.025821Z","shell.execute_reply.started":"2023-11-26T08:52:52.524329Z","shell.execute_reply":"2023-11-26T08:52:53.024866Z"}}
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:53.026984Z","iopub.execute_input":"2023-11-26T08:52:53.027259Z","iopub.status.idle":"2023-11-26T08:52:53.032699Z","shell.execute_reply.started":"2023-11-26T08:52:53.027235Z","shell.execute_reply":"2023-11-26T08:52:53.031913Z"}}
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Totale', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:53.033875Z","iopub.execute_input":"2023-11-26T08:52:53.034149Z","iopub.status.idle":"2023-11-26T08:52:53.045234Z","shell.execute_reply.started":"2023-11-26T08:52:53.034127Z","shell.execute_reply":"2023-11-26T08:52:53.044455Z"}}
def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:53.046950Z","iopub.execute_input":"2023-11-26T08:52:53.047370Z","iopub.status.idle":"2023-11-26T08:52:53.060421Z","shell.execute_reply.started":"2023-11-26T08:52:53.047321Z","shell.execute_reply":"2023-11-26T08:52:53.059724Z"}}
def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:53.061444Z","iopub.execute_input":"2023-11-26T08:52:53.061720Z","iopub.status.idle":"2023-11-26T08:52:53.139107Z","shell.execute_reply.started":"2023-11-26T08:52:53.061693Z","shell.execute_reply":"2023-11-26T08:52:53.138441Z"}}
def set_color_map(color_list):
    cmap_custom = ListedColormap(color_list)
    print("Notebook Color Schema:")
    sns.palplot(sns.color_palette(color_list))
    plt.show()
    return cmap_custom

color_list = ['#C5FFF8', '#96EFFF', '#5FBDFF', '#7B66FF']
cmap_custom = set_color_map(color_list)


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:52:53.140215Z","iopub.execute_input":"2023-11-26T08:52:53.140642Z","iopub.status.idle":"2023-11-26T08:52:53.148376Z","shell.execute_reply.started":"2023-11-26T08:52:53.140615Z","shell.execute_reply":"2023-11-26T08:52:53.147503Z"}}
def plot_count_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette= ['#C5FFF8', '#96EFFF', '#5FBDFF', '#7B66FF'])
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title}")
    plt.show()    
def plot_distribution_pairs(data_df, feature, title, hue="set"):
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(data_df[hue].unique()):
        g = sns.histplot(data_df.loc[data_df[hue]==h, feature], color=color_list[i], ax=ax, label=h)
    #plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {title}")
    g.legend()
    plt.show()

# %% [code]
def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = split_text[0] + "."
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
