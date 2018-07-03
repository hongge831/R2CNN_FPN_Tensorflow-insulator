import os,sys
from tqdm import tqdm
cur_dir = '/home/tanghm/Documents/YFF/R2CNN_FPN_Tensorflow-master/data/VOCdevkit_train/JPEGImages'
for filename in tqdm(os.listdir(cur_dir)):  
    file_ext = os.path.splitext(filename)[1]  
    if file_ext == '.JPG':  
        newfile = filename.replace(file_ext, '.jpg')  
        os.rename(cur_dir+'/'+filename, cur_dir+'/'+newfile)  