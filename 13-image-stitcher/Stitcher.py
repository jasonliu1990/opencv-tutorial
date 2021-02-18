import cv2
import numpy as np


class Stitcher:

    # 拼接函數
    def stitch(self, images, ratio=0.75, reporj_thresh=4.0, show_matches=False):
        # input img
        (image_b, image_a) = images
        # 檢測A, B圖片的SIFT關鍵特徵點並計算特徵描述子
        (kps_a, features_a) = self.detect_and_describe(image_a)
        (kps_b, features_b) = self.detect_and_describe(image_b)
        # 匹配兩張圖片的特徵點
        M = self.match_keypoints(kps_a, kps_b, features_a, features_b, ratio, reporj_thresh)
        # 如果為空就退出
        if M is None:
            return None

        # 提取匹配結果, H是3*3視角變換矩陣
        (matches, H, status) = M
        # 圖片A視角轉換
        result = cv2.warpPerspective(image_a, H, (image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
        # self.cv_show('result', result)
        # 圖片B傳入result圖片的左端
        result[0: image_b.shape[0], 0:image_b.shape[1]] = image_b

        # 檢測是否需要顯示圖片匹配
        if show_matches:
            vis = self.draw_matches(image_a, image_b, kps_a, kps_b, matches, status)
            return (result, vis)

        return result

    def cv_show(self,name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_and_describe(self, image):
        # 灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # SIFT生成
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 檢測SIFT特徵並計算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)
        # 轉成np
        kps = np.float32([kp.pt for kp in kps])

        return (kps, features)
    # 暴力匹配器
    def match_keypoints(self, kps_a, kps_b, features_a, features_b, ratio, reporj_thresh):
         matcher = cv2.BFMatcher()
         # 用KNN 檢測AB圖的SIFT特徵匹配對
         raw_maches = matcher.knnMatch(features_a, features_b, 2)
         matches = []
         for m in raw_maches:
             # 當最近距離與次近距離的比值小於ratio就保留
             if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                 matches.append((m[0].trainIdx, m[0].queryIdx))
         # 匹配對數>4時, 計算視角變換矩陣
         if len(matches) > 4:
             pts_a = np.float32([kps_a[i] for (_, i) in matches])
             pts_b = np.float32([kps_b[i] for (i, _) in matches])
             (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reporj_thresh)

             return (matches, H, status)
         # 匹配對數<4時, 返回None
         return None

    def draw_matches(self, image_a, image_b, kps_a, kps_b, matches, status):
        # 初始化, 將AB圖連接
        (hA, wA) = image_b.shape[:2]
        (hB, wB) = image_b.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = image_a
        vis[0:hB, wA:] = image_b

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kps_a[queryIdx][0]), int(kps_a[queryIdx][1]))
                ptB = (int(kps_b[trainIdx][0]) + wA, int(kps_b[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis








