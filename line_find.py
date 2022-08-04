#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time

Width = 640
Height = 480
Offset=330

Gap = 40
prev_left=90
prev_right=580
prev_left_tilt=1
prev_right_tilt=-1

#draw ROI
def draw_roi(img):
    img=cv2.line(img, (0, Offset), (img.shape[1], Offset), (229, 255, 207), 4)
    img=cv2.line(img, (0, Offset), (0, Offset+Gap), (229, 255, 207), 4)
    img=cv2.line(img, (img.shape[1], Offset+Gap), (0, Offset+Gap), (229, 255, 207), 4)
    img=cv2.line(img, (img.shape[1], Offset), (img.shape[1], Offset+Gap), (229, 255, 207), 4)
    return img
        
# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)
        
        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = float(x_sum) / float(size * 2)
    y_avg = float(y_sum) / float(size * 2)

    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)
    
    x1, x2 = 0, 0
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

    return x1, x2, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    global prev_left, prev_right, prev_left_tilt, prev_right_tilt
    offset=Offset
    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge = cv2.Canny(np.uint8(gray), low_threshold, high_threshold)

    # HoughLinesP
    roi = edge[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,30,10)

    # divide left, right lines
    if all_lines is None:
        return (0, 640), frame

    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    lx1, lx2, lpos = get_line_pos(left_lines, left=True)
    rx1, rx2, rpos = get_line_pos(right_lines, right=True)
    
    # draw lines
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)
    
    cv2.rectangle(frame, (image.shape[1]/2-5, 15 + offset),(image.shape[1]/2+5, 25 + offset),(0, 0, 255), 2)

    if(abs((lx2-lx1)/Height)>abs(prev_left_tilt)+0.5):
        lpos=prev_left
    if(abs(rx2-rx1)/Height>abs(prev_right_tilt)+0.5):
        rpos=prev_right

    if(lpos>0 and rpos>=640):
        frame = cv2.line(frame, (int(lx1), Height), (int(lx2), (Height/2)), (255, 0,0), 3)
        cv2.rectangle(frame, (lpos - 5, 15 + offset),(lpos + 5, 25 + offset),(0, 255, 0), 2)
        cv2.rectangle(frame, ((lpos+prev_right)/2-5, 15 + offset),((lpos+prev_right)/2+5, 25 + offset),(0, 255, 0), 2)    


    elif(lpos<=0 and rpos<640):
        frame = cv2.line(frame, (int(rx1), Height), (int(rx2), (Height/2)), (255, 0,0), 3)
        cv2.rectangle(frame, (rpos - 5, 15 + offset),(rpos + 5, 25 + offset),(0, 255, 0), 2)
        cv2.rectangle(frame, ((prev_left+rpos)/2-5, 15 + offset),((prev_left+rpos)/2+5, 25 + offset),(0, 255, 0), 2)    

    elif(lpos<=0 and rpos>=640):
        lpos=prev_left
        rpos=prev_right
    else:
        frame = cv2.line(frame, (int(lx1), Height), (int(lx2), (Height/2)), (255, 0,0), 3)
        frame = cv2.line(frame, (int(rx1), Height), (int(rx2), (Height/2)), (255, 0,0), 3)
        cv2.rectangle(frame, (lpos - 5, 15 + offset),(lpos + 5, 25 + offset),(0, 255, 0), 2)
        cv2.rectangle(frame, (rpos - 5, 15 + offset),(rpos + 5, 25 + offset),(0, 255, 0), 2)
        cv2.rectangle(frame, ((lpos+rpos)/2-5, 15 + offset),((lpos+rpos)/2+5, 25 + offset),(0, 255, 0), 2)    

    prev_left=lpos
    prev_right=rpos
    prev_left_tilt=(lx2-lx1)/Height
    prev_right_tilt=(rx2-rx1)/Height
    return (lpos, rpos), frame

def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728

    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 1.5, 0.7) #1.0?   
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    cv2.imshow('steer', image)

# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(1)

    while not rospy.is_shutdown():
        ret, image = cap.read()

        pos, frame = process_image(image)
        image=draw_roi(image)

        distance = frame.shape[1]/2-(pos[0]+pos[1])/2
        steer_angle = math.atan2(distance, 110)*180/math.pi
        steer_angle = steer_angle*0.4
        draw_steer(frame, steer_angle)
            
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
