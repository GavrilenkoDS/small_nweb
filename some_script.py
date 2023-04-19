import pandas as pd
import numpy as np 
import os 
import sys
import shutil

def get_pd_df(root):
    df = pd.read_csv(root)
    return df 


mydf = get_pd_df(r"C:\Users\Dmitrii\Desktop\neuro_data\test.csv")

print (mydf)

# df_w_label_0 = mydf[mydf['label']==0]
# df_w_label_1 = mydf[mydf['label']==1]


# piclist_w_label_0 = df_w_label_0['name']
# piclist_w_label_1 = df_w_label_1['name']


# piclist_w_label_0_mass = piclist_w_label_0.to_numpy()
# piclist_w_label_1_mass = piclist_w_label_1.to_numpy()

pics = mydf['name']
pics.to_numpy()

print (pics)
#machine - 0
#hand - 1


rootpathpic = r"C:\Users\Dmitrii\Desktop\neuro_data\imgs"
trainpathmachine = r"C:\Users\Dmitrii\Desktop\neuro_data\test"



for name in pics:
    shutil.copy2(os.path.join(rootpathpic,name),os.path.join(trainpathmachine,name))

# for name in piclist_w_label_1_mass:
#     shutil.copy2(os.path.join(rootpathpic,name),os.path.join(trainpathhand,name))

