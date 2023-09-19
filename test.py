import cv2
tmp_path = "../supplementary_experiment/rotated_img/template.bmp"
tmp_img = cv2.imread(tmp_path, 0)
print(tmp_img.shape)