#-*- coding: utf-8 -*-
import numpy as np
import cv2
import imutils
from imutils import contours
import myutils
import argparse
# config
# template_path = './images/template_img/ocr_a_reference.png'
# img_path = './images/test_img'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# card type
FIRST_NUMBER = {
    '3': 'American Express',
    '4': 'Visa',
    '5': 'Master card',
    '6': 'Discover card'
}

# def function
def cv_show(img, name='image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# read template
img = cv2.imread(args['template'])
cv_show(img)
# 灰階
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(ref)
# 二值
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show(ref)
# 輪廓檢測
ref_cnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 只檢測外輪廓
cv2.drawContours(img, ref_cnts, -1, (0, 0, 255), 3)
cv_show(img)
print(np.array(ref_cnts).shape)
# 排序
ref_cnts = myutils.sort_contours(ref_cnts, method='left-to-right')[0]
digits = {}
# 遍歷輪廓
for (i, c) in enumerate(ref_cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y: y+h, x: x+w]
    roi = cv2.resize(roi, (57, 88))
    # 每個數字對應一個模板
    digits[i] = roi

# 對輸入圖像操作
# 初始化捲積核
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 圖像輸入及預處理
image = cv2.imread(args['image'])
cv_show(image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show(gray)
# 禮帽, 凸顯明亮的區域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
cv_show(tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(min_val, max_val) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - min_val) / (max_val - min_val)))
gradX = gradX.astype('uint8')
print(np.array(gradX).shape)
cv_show(gradX)

# 閉操作, 將數字連在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kernel)
cv_show(gradX)
# THRESH_OTSU會自動尋找合適的閾值
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show(thresh)
# again
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel) 
cv_show(thresh)

# 計算輪廓
thresh_cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 只檢測外輪廓
cnts = thresh_cnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
cv_show(cur_img)

locs = []

# 遍歷輪廓
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 選擇合適的區域
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h <20):
            locs.append((x, y, w, h))
# 左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    group_output = []

    # 依據座標取出每一組
    group = gray[gY - 5: gY + gH + 5, gX - 5: gX + gW + 5]
    cv_show(group)
    # 預處理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show(group)
    # 每一組輪廓
    digits_cnt, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_cnt = contours.sort_contours(digits_cnt, method='left-to-right')[0]
    # 計算每一個數值
    for c in digits_cnt:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y: y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show(roi)
        # 計算匹配得分
        scores = []
        # 在模板中計算得分
        for (digit, digit_roi) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digit_roi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 得到最適合的數字
        group_output.append(str(np.argmax(scores)))

    # plot
    cv2.rectangle(image, (gX - 5, gY -5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv2.putText(image, ''.join(group_output), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    output.extend(group_output)

print(f'credit card type {FIRST_NUMBER[output[0]]}')
print(f"credit card #: {''.join(output)}")
cv_show(image)






















