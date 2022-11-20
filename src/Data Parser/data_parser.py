import pandas as pd
import re

pd.set_option('display.max_colwidth', None)

df1 = pd.read_csv("./attributes/image_attribute_labels.txt", sep=" ", header=None, names=["Image_ID", "Attribute_ID", "Is_Present", "NA1", "NA2"])
df3 = pd.read_csv("images.txt", sep=" ", header=None, names=["Image_ID", "Image_Name"])
df4 = pd.read_csv("attributes.txt", sep=" ", header=None, names=["Attribute_ID"])
df4 = df4.T
#print(df4.tail(1))

selected_cols = df1[["Image_ID","Attribute_ID","Is_Present"]]
df2 = selected_cols.copy()
df2['Is_Present'] = df2['Is_Present'].replace(0,-1)
df2["Image_Name"] = df3["Image_Name"]
#df2["Attribute_Name"] = df4["Attribute_Name"]
#print(df2.head(10))
#print(df4.head(10))

df_final = df2[["Image_ID","Image_Name","Attribute_ID", "Is_Present"]]
#df_final = df_final.drop(["Image_ID", "Attribute_Name"],axis=1)
df_final = df_final.pivot(index=['Image_ID'],columns='Attribute_ID',values=['Is_Present'])
df_final.columns = df_final.columns.droplevel()
df_final=df_final.rename_axis(None,axis=1)
df_final = df_final.reset_index()
#print(modified_df.head(5))
#print(df_final.head(5))
#print("df_final.columns.values: ")
#print(df_final.columns.values)
#print(len(df_final['Image_ID'].unique().tolist()))
#print("len(df_final): ")
#print(len(df_final))
#print(len(df_final.Image_ID.unique()))

df_final = df_final.join(df3.set_index('Image_ID'),on='Image_ID')
#print(df_final.head(5))
#print("df_final.columns.values: ")
#print(df_final.columns.values)
#print("len(df_final): ")
#print(len(df_final))

Complete_df = df_final[['Image_Name',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312]]
#modified_df = modified_df.drop(['Image_ID'],axis=1)
modified_df = Complete_df[['Image_Name',249,250,260,254,261,52,45,40,51,41,37,36,30,25,26]]
print(modified_df.head(5))
print("len(modified_df): ")
print(len(modified_df))
print("modified_df.columns.values: ")
print(modified_df.columns.values)

#train_df = modified_df.iloc[:11700]
#val_df = modified_df.iloc[11700:11760]
#test_df = modified_df.iloc[11760:]

#modified_df.to_csv(r'C:\Users\itish\OneDrive\Documents\GitHub\Style-AttnGAN\data\birds\CUB_200_2011\parsed_data.txt', header=None, index=None, sep=' ', mode='a')
modified_df.to_csv(r'C:\Users\itish\OneDrive\Documents\GitHub\Style-AttnGAN\data\birds\CUB_200_2011\train_label.txt', header=None, index=None, sep=' ', mode='a')
#val_df.to_csv(r'C:\Users\itish\OneDrive\Documents\GitHub\Style-AttnGAN\data\birds\CUB_200_2011\val_label.txt', header=None, index=None, sep=' ', mode='a')
#test_df.to_csv(r'C:\Users\itish\OneDrive\Documents\GitHub\Style-AttnGAN\data\birds\CUB_200_2011\test_label.txt', header=None, index=None, sep=' ', mode='a')