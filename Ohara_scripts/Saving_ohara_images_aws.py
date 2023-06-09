import pandas as pd
import glob
df_urls = pd.read_csv('Tasmania_ohara_labels/Ohara_all_images_labels/collection-u1578-Ohara_transect_07_dataset-9702-1987c9e43fa3c96d4373-dataframe.csv')



exist_images = glob.glob('Tasmania_ohara_labels/Ohara_all_images_labels/image_labels/*')
df_exist = pd.DataFrame(exist_images, columns =['key'])
df_exist['key'] = df_exist['key'].str.split('/').str[-1]
#perform outer join
outer = df_urls.merge(df_exist, how='outer', indicator=True)
#perform anti-join
anti_join = outer[(outer._merge=='left_only')].drop('_merge', axis=1)

df_in = anti_join.reset_index()


import requests # request img from web
import shutil # save img locally
def save_image_from_url(df, idx):
    url = df['path_best'][idx] 
    path = 'Tasmania_ohara_labels/Ohara_all_images_labels/image_labels/'
    file_name = path + df['key'][idx]

    res = requests.get(url, stream = True)

    if res.status_code == 200:
        with open(file_name,'wb') as f:
            shutil.copyfileobj(res.raw, f)
        #print('Image sucessfully Downloaded: ',file_name)
    else:
        print('Image Couldn\'t be retrieved')
        
for i in range(len(df_in)):  
    save_image_from_url(df_in, i)
    if i%400==0:
        print(i)