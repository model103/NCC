import cv2
import time
import os

import numpy as np

src_paths = "../supplementary_experiment/rotated_img/rotated/"
tmp_path = "../supplementary_experiment/rotated_img/template.bmp"
save_path = "../supplementary_experiment/rotated_img/"

tmp_img = cv2.imread(tmp_path, 0)
th, tw = tmp_img.shape

src_images = os.listdir(src_paths)
Note = open(save_path+'result.txt', mode='w')
line = "name    " + 'col    ' + 'row    ' + 'score   ' + 'time  ' +"\n"
Note.writelines(line)
times = []
col = []
row = []
for img_name in src_images:  #以纯数字命名的图片
    src_img = cv2.imread(src_paths+img_name, 0)
    start = time.time()
    res = cv2.matchTemplate(tmp_img, src_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    target_col = max_loc[0]+tw/2  #检出的目标的中心位置
    target_row = max_loc[1]+th/2
    end = time.time()
    total_time = float(end - start)*1000
    times.append(total_time)
    print(img_name, " col：", target_col," row：", target_row, "  耗时：", total_time, "ms", " score:", max_val)
    #保存结果
    line = img_name + "   " + str(target_col) + "   " + str(target_row) + "   " + str(max_val) + "   " + str(total_time) + "\n"
    col.append(target_col)
    row.append(target_row)
    Note.writelines(line)
    # 将匹配结果框起来
    #tl = max_loc
    #br = (tl[0] + tw, tl[1] + th)
    #cv2.rectangle(src_img, tl, br, (0, 0, 255), 2)
    #cv2.imwrite(save_path+src_path, src_img)
print("时间均值", np.mean(times), "标准差：", np.std(times))
Note.close()

'''
#计算检测精度，平均误差，std
ture_col= np.array([269
,365
,360.3014256
,346.6656315
,325.4273842
,298.6656315
,269
,239.3343685
,212.5726158
,191.3343685
,177.6985744
,173
,177.6985744
,191.3343685
,212.5726158
,239.3343685
,269
,298.6656315
,325.4273842
,346.6656315
,360.3014256
])
ture_row = np.array([190
,190
,219.6656315
,246.4273842
,267.6656315
,281.3014256
,286
,281.3014256
,267.6656315
,246.4273842
,219.6656315
,190
,160.3343685
,133.5726158
,112.3343685
,98.69857444
,94
,98.69857444
,112.3343685
,133.5726158
,160.3343685
])

error = np.sqrt((ture_row - np.array(row))**2 + (ture_col - np.array(col))**2)
print('mean_error', np.mean(error))
print('std_error', np.std(error))

'''


