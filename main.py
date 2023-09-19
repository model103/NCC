import math
import numpy as np
import cv2
import time
import os
from PyQt5.QtCore import QTime

'''
1.二维数组降维
2.圆投影匹配算法
'''


def calculate_unrotate_temp_data(temp):
    ####使用圆投影匹配算法####
    temp_mean = np.mean(temp)
    temp_sub_avg = temp - temp_mean
    temp_deviation = np.vdot(temp_sub_avg, temp_sub_avg)
    return temp_deviation, temp_sub_avg


def calculate_rotate_temp_data(temp):
    temp_column = temp.shape[1]
    temp_row = temp.shape[0]
    if temp_column < temp_row:
        diameter = temp_row
    else:
        diameter = temp_column
    max_radius = math.floor(diameter / 2)
    circle_center = (temp_row / 2, temp_column / 2)
    circle_ring_point = {}

    ###统计每个点到中心的半径，并分类###
    for i in range(temp_column):
        for j in range(temp_row):
            radius = round(np.sqrt((i - circle_center[0]) ** 2 + (j - circle_center[1]) ** 2))
            if radius > max_radius:
                continue
            if radius in circle_ring_point.keys():
                circle_ring_point[radius].append(j * temp_column + i)
            else:
                circle_ring_point[radius] = [j * temp_column + i]

    ###排序获取每个环上的点###
    circle_ring_point = sorted(circle_ring_point.items(), key=lambda item: item[0])
    circular_projection_data = []
    for item in circle_ring_point:
        circular_projection_data.append(np.array(item[1]))

    _circle_sum = []
    _temp = temp.reshape(1, -1)[0]
    for item in circular_projection_data:
        _circle_sum.append(np.sum(_temp[item]))
    _circle_sum = np.array(_circle_sum)
    _mean = np.mean(_circle_sum)
    _deviation_array = _circle_sum - _mean
    _deviation = np.dot(_deviation_array, _deviation_array)

    tempData = {'deviation': _deviation, 'deviation_array': _deviation_array,
                'circular_projection_data': circular_projection_data, 'temp_size': temp.shape}

    return tempData


def generate_temp_data(temp, downsamplingtime=0, is_rotate=False):
    ######每次从原图开始取样#############
    temp_downsampling_data = []
    temp_downsampling_img = []

    ###generate downsampling img###
    temp_downsampling_img.append(temp)
    for i in range(downsamplingtime):
        temp_downsampling_img.append(cv2.pyrDown(temp_downsampling_img[i]))

    ###generate downsampling data###
    for temp_img in temp_downsampling_img:
        if is_rotate:
            temp_downsampling_data.append(calculate_rotate_temp_data(temp_img))
        else:
            temp_downsampling_data.append(
                {'deviation': (calculate_unrotate_temp_data(temp_img))[0],
                 'sub_avg': (calculate_unrotate_temp_data(temp_img))[1]})
    return temp_downsampling_data


def ncc_unrotate_match(src, temp_data, threshold=0.5, match_region=None):
    temp_deviation, temp_sub_avg = temp_data['deviation'], temp_data['sub_avg']
    temp_row_num = temp_sub_avg.shape[0]
    temp_column_num = temp_sub_avg.shape[1]

    _line_start = 0
    _column_start = 0
    _line_range = src.shape[0] - temp_row_num + 1
    _column_range = src.shape[1] - temp_column_num + 1
    if match_region is not None:
        _line_start = match_region[1]
        _column_start = match_region[0]
        _line_range = match_region[1] + match_region[3] + 1
        _column_range = match_region[0] + match_region[2] + 1
        if _line_range > src.shape[0] - temp_row_num + 1:
            _line_range = src.shape[0] - temp_row_num + 1
        if _column_range > src.shape[1] - temp_column_num + 1:
            _column_range = src.shape[1] - temp_column_num + 1

    src_integration = cv2.integral(src)
    pixel_num = temp_sub_avg.size
    match_points = []
    for i in range(_line_start, _line_range, 1):
        for j in range(_column_start, _column_range, 1):
            src_mean = (src_integration[i + temp_row_num][j + temp_column_num] +
                        src_integration[i][j] -
                        src_integration[i][j + temp_column_num] -
                        src_integration[i + temp_row_num][
                            j]) / pixel_num
            _src_deviation = src[i:i + temp_row_num, j:j + temp_column_num] - src_mean
            src_deviation = np.vdot(_src_deviation, _src_deviation)

            ncc_numerator = np.vdot(temp_sub_avg, _src_deviation)

            ncc_denominator = np.sqrt(temp_deviation * src_deviation)
            ncc_value = ncc_numerator / ncc_denominator
            if ncc_value > threshold:
                match_point = {'match_score': ncc_value, 'point': (j, i)}
                match_points.append(match_point)
            #print("match_point = ", match_point)    ####################
    return match_points


def ncc_rotate_match(src, tempData, threshold=0.5, angle_start=0, angle_end=360, angle_step=1, match_region=None):
    temp_deviation = tempData['deviation']
    temp_deviation_array = tempData['deviation_array']
    circular_projection_data = tempData['circular_projection_data']

    temp_row_num, temp_column_num = tempData['temp_size'][0], tempData['temp_size'][1]
    _line_start, _column_start, _line_range, _column_range = 0, 0, src.shape[0] - temp_row_num, src.shape[
        1] - temp_column_num

    if match_region is not None:
        _line_start = match_region[1]
        _column_start = match_region[0]
        _line_range = match_region[1] + match_region[3] + 1
        _column_range = match_region[0] + match_region[2] + 1
        if _line_range > src.shape[0] - temp_row_num + 1:
            _line_range = src.shape[0] - temp_row_num + 1
        if _column_range > src.shape[1] - temp_column_num + 1:
            _column_range = src.shape[1] - temp_column_num + 1

    match_points = []
    for i in range(_line_start, _line_range, 1):
        for j in range(_column_start, _column_range, 1):
            _src = src[i:i + temp_row_num, j:j + temp_column_num].reshape(1, -1)[0]
            src_sum = []
            for item in circular_projection_data:
                src_sum.append(np.sum(_src[item]))
            _src_sum = np.array(src_sum)
            src_mean = np.mean(_src_sum)
            src_deviation_array = _src_sum - src_mean

            ncc_numerator = np.vdot(src_deviation_array, temp_deviation_array)

            src_deviation = np.dot(src_deviation_array, src_deviation_array)
            ncc_denominator = np.sqrt(temp_deviation * src_deviation)
            ncc_value = ncc_numerator / ncc_denominator
            if ncc_value > threshold:
                match_point = {'match_score': ncc_value, 'point': (j, i)}
                match_points.append(match_point)
            print("match_point = ", match_point)   #########################
    return match_points


def ncc_match(src, temp, is_rotate=False, downsamplingtime=0, threshold=0.7, angle_start=0, angle_end=0,
              match_region=None):
    assert temp.shape[0] <= src.shape[0] and temp.shape[1] <= src.shape[1]

    temp_downsampling_data = generate_temp_data(temp, downsamplingtime, is_rotate)

    src_down_sampling_array = []
    src_down_sampling_array.append(src)
    for i in range(1, downsamplingtime + 1):
        src_down_sampling_array.append(cv2.pyrDown(src_down_sampling_array[i - 1]))

    match_points = []
    downsample_match_point = None
    for i in range(downsamplingtime, -1, -1):
        match_offset = 2 ** (i + 1)
        if i == downsamplingtime:
            match_region = [0, 0, src_down_sampling_array[i].shape[1], src_down_sampling_array[i].shape[0]]
        else:
            _x, _y, _w, _h = 0, 0, 0, 0
            if downsample_match_point[0] * 2 - match_offset >= 0:
                _x = downsample_match_point[0] * 2 - match_offset
                _w = match_offset * 2 + 1
            else:
                _x = 0
                _w = match_offset + 1
            if downsample_match_point[1] * 2 - match_offset >= 0:
                _y = downsample_match_point[1] * 2 - match_offset
                _h = match_offset * 2 + 1
            else:
                _y = 0
                _h = match_offset + 1
            match_region = [_x, _y, _w, _h]

        if not is_rotate:
            _match_points = ncc_unrotate_match(src_down_sampling_array[i],
                                               temp_downsampling_data[i], match_region=match_region,
                                               threshold=threshold)
        else:
            _match_points = ncc_rotate_match(src_down_sampling_array[i],
                                             temp_downsampling_data[i], match_region=match_region,
                                             threshold=threshold)
        if i == 0:
            match_points = _match_points
        if len(_match_points) != 0:
            ###利用上一层的最佳匹配值来作为下一层匹配的种子点###
            downsample_match_point = sorted(_match_points, key=lambda _point: _point['match_score'], reverse=True)[0][
                'point']
        else:
            break
    return match_points


def draw_result(src, temp, match_point):
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(src, match_point,
                  (match_point[0] + temp.shape[1], match_point[1] + temp.shape[0]),
                  (0, 255, 0), 1)
    cv2.imshow('temp', temp)
    cv2.imshow('result', src)
    cv2.waitKey()


if __name__ == '__main__':
    src_paths = "../paper_images_channel3/Occlusion/"
    src_images = os.listdir(src_paths)
    temp = cv2.imread('../paper_images_channel3/Occlusion.bmp', cv2.IMREAD_GRAYSCALE)

    downsamplingtime = 4
    threshold = 0
    is_rotate = False

    for src_image in src_images:
        src_path = src_paths + src_image
        src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        start  = time.time()
        match_points = ncc_match(src, temp, is_rotate=is_rotate, threshold=threshold, downsamplingtime=downsamplingtime)
        if len(match_points) != 0:
            best_match_point = sorted(match_points, key=lambda _point: _point['match_score'], reverse=True)[0]
            print(best_match_point)    #显示的是检测框左上角位置
            end = time.time()
            print("耗时：", (end - start)*1000, "ms")
            #draw_result(src, temp, best_match_point['point'])
        else:
            print("no match point")
