import torch
import traceback
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from numpy import ndarray
import cv2
import onnxruntime
import logging
from skimage import transform as trans
from concurrent.futures import ThreadPoolExecutor
import os
import faiss


def get_logger(name='_main_', log_file='./logs/eval.log', log_level=logging.DEBUG):
    # 定义文件路径
    file_path = 'logs'
    # 确保文件所在的目录存在，如果不存在则创建
    os.makedirs(file_path, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


logger = get_logger()


# 创建一个日志记录器实例，并设置其日志级别为 INFO。
# 创建一个 FileHandler 来处理日志文件的写入，并设置其日志级别为 INFO。
# 定义一个日志格式，包括时间戳、日志记录器名称、日志级别和日志消息。
# 将定义的日志格式应用到文件处理器上。
# 将文件处理器添加到日志记录器上。
# 创建一个 StreamHandler 来处理日志的输出到标准输出（通常是控制台），并设置其日志级别为 INFO。
# 将相同的日志格式应用到流处理器上。
# 将流处理器添加到日志记录器上。
# 返回配置好的日志记录器实例。


# 这段代码定义了一个名为 estimate_norm 的函数，它用于估计给定的面部关键点（landmarks）的归一化变换矩阵
def estimate_norm(lmk, image_size=112, mode='arcface'):
    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)

    # 定义了arcface模式下的标准坐标arcface_src
    arcface_src = np.expand_dims(arcface_src, axis=0)

    # 检查输入的lmk形状是否正确（5个关键点，每个关键点2个坐标）
    assert lmk.shape == (5, 2)
    # 创建一个trans.SimilarityTransform对象tform，用于估计相似性变换
    tform = trans.SimilarityTransform()
    # 将lmk的每个关键点坐标扩展为三维数组，添加一个值为1的第三维
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    # 根据mode和image_size选择使用arcface_src还是src_map中的标准坐标
    if mode == 'arcface':
        # assert image_size == 112
        if image_size != 112:
            src = arcface_src * (image_size / 112.0)
        else:
            src = arcface_src
    else:
        # src = src_map[image_size]
        pass
    # 遍历每个标准坐标，使用tform.estimate方法估计变换矩阵M
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        # 计算变换后的关键点与标准坐标之间的误差，并记录误差最小的变换矩阵和对应的标准坐标索引
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    # 最后，函数返回误差最小的变换矩阵min_M和对应的标准坐标索引min_index
    return min_M, min_index


# 这段代码定义了一个名为 norm_crop 的函数，它用于根据给定的面部关键点进行人脸的归一化裁剪。
def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
# 调用 estimate_norm 函数来估计给定关键点的归一化变换矩阵 M 和姿势索引 pose_index。
# 使用 cv2.warpAffine 函数将输入图像 img 通过变换矩阵 M 进行仿射变换，输出尺寸为 (image_size, image_size) 的图像。
# 返回变换后的图像 warped


# 这段代码定义了一个名为 distance2bbox 的函数，它用于将距离预测转换为边界框（bounding box）
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def select_face(det, mode="combined", img_shape=None, prob_factor=0.8):
    best_i = 0
    if mode == "longst":
        max_area = (det[0, 2] - det[0, 0]) * (det[0, 2] - det[0, 0]) + (det[0, 3] - det[0, 1]) * (det[0, 3] - det[0, 1])
        for i in range(1, len(det)):
            this_area = (det[i, 2] - det[i, 0]) * (det[i, 2] - det[i, 0]) + (det[i, 3] - det[i, 1]) * (det[i, 3] - det[i, 1])
            if this_area > max_area:
                max_area = this_area
                best_i = i
        return best_i
    elif mode == "largest":
        rect = det[0]
        top_left_x, top_left_y = max(0, rect[0]), max(0, rect[1])
        bottom_right_x, bottom_right_y = min(rect[2], img_shape[1]), min(rect[3], img_shape[0])
        max_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
        for i in range(1, len(det)):
            rect = det[i]
            top_left_x, top_left_y = max(0, rect[0]), max(0, rect[1])
            bottom_right_x, bottom_right_y = min(rect[2], img_shape[1]), min(rect[3], img_shape[0])
            this_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
            if this_area > max_area:
                max_area = this_area
                best_i = i
        return best_i
    elif mode == "combined":
        rect = det[0]
        top_left_x, top_left_y = max(0, rect[0]), max(0, rect[1])
        bottom_right_x, bottom_right_y = min(rect[2], img_shape[1]), min(rect[3], img_shape[0])
        max_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
        for i in range(1, len(det)):
            rect = det[i]
            top_left_x, top_left_y = max(0, rect[0]), max(0, rect[1])
            bottom_right_x, bottom_right_y = min(rect[2], img_shape[1]), min(rect[3], img_shape[0])
            this_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
            if this_area > max_area:
                max_area = this_area
                best_i = i

        rect = det[best_i]
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = rect[0], rect[1], rect[2], rect[3]
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2
        best_dis = (center_x - img_shape[1] / 2) ** 2 + (center_y - img_shape[0] / 2) ** 2
        best_prob = rect[4]
        for i in range(len(det)):
            if i == best_i:
                continue
            this_area = (det[i, 2] - det[i, 0]) * (det[i, 2] - det[i, 0]) + (det[i, 3] - det[i, 1]) * (det[i, 3] - det[i, 1])
            this_prob = det[i, 4]
            if this_area < max_area * 0.6:
                continue
            if this_prob < best_prob * prob_factor:
                continue
            rect = det[i]
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = rect[0], rect[1], rect[2], rect[3]
            center_x = (top_left_x + bottom_right_x) / 2
            center_y = (top_left_y + bottom_right_y) / 2
            this_dis = (center_x - img_shape[1] / 2) ** 2 + (center_y - img_shape[0] / 2) ** 2
            if this_dis < best_dis:
                best_dis = this_dis
                best_i = i
        return best_i


# 这段代码定义了一个名为 PyFAT 的类，它用于加载和管理一个基于 ONNX 模型的面部识别系统。
# 这个类包含了初始化方法 __init__ 和一个 load 方法，用于加载检测模型和特征提取模型。
class PyFAT:
    def __init__(self, N: int, K: int) -> None:
        self.crop = None
        self.nms_thresh = 0.4
        self.use_kps = None
        self._num_anchors = None
        self._feat_stride_fpn = None
        self.fmc = None
        self._anchor_ratio = None
        self.batched = None
        self.output_names = None
        self.label_lib = None
        self.idx_lib = None
        self.N = N
        self.K = K
        self.detect_model = None
        self.model = None
        self.feature_len = None
        self.feat_lib = None
        self.feat_lib_list = []
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache = {}
        self.thread_num = 1
        self.idx_map = []
        # self.gallery = faiss.IndexFlatIP(512)
        self.gallery_list = []
        self.index = 0
        # self.gallery_usable_list = []
        self.gallery_usable_dict = {}

    # 这段代码是PyFAT类中的load方法，它的作用是加载两个ONNX模型：一个检测模型和一个特征提取模型，并根据模型输出配置类的属性。
    def load(self, assets_path='./assets/', devices=[0]) -> int:
        assets_path = Path(assets_path)
        rec_model_path = assets_path / '0913_backbone_ep0000.onnx'
        det_model_path = assets_path / 'det_10g.onnx'
        try:
            providers = [
                ('CUDAExecutionProvider', {'device_id': 1}),
                'CPUExecutionProvider'
            ]
            self.detect_model = onnxruntime.InferenceSession(det_model_path, providers=providers)
            self.model = onnxruntime.InferenceSession(rec_model_path, providers=providers)
            self.model_input_name = self.model.get_inputs()[0].name

            outputs = self.detect_model.get_outputs()
            if len(outputs[0].shape) == 3:
                self.batched = True
            output_names = []
            for o in outputs:
                output_names.append(o.name)
            self.output_names = output_names
            self.input_mean = 127.5
            self.input_std = 128.0
            self.use_kps = False
            self._anchor_ratio = 1.0
            self._num_anchors = 1
            if len(outputs) == 6:
                self.fmc = 3
                self._feat_stride_fpn = [8, 16, 32]
                self._num_anchors = 2
            elif len(outputs) == 9:
                self.fmc = 3
                self._feat_stride_fpn = [8, 16, 32]
                self._num_anchors = 2
                self.use_kps = True
            elif len(outputs) == 10:
                self.fmc = 5
                self._feat_stride_fpn = [8, 16, 32, 64, 128]
                self._num_anchors = 1
            elif len(outputs) == 15:
                self.fmc = 5
                self._feat_stride_fpn = [8, 16, 32, 64, 128]
                self._num_anchors = 1
                self.use_kps = True

            return 0
        except Exception:
            traceback.print_exc()
            logger.info("模型加载失败")
            return -1

    def unload(self) -> None:
        pass

    def get_feature_parallel_num(self) -> Tuple[int, int]:
        logger.info('in get_feature_parallel_num')
        # return self.thread_num, 8
        return 1, 32

    def get_topk_parallel_num(self) -> Tuple[int, int]:
        logger.info('in get_topk_parallel_num')
        # return self.thread_num, 8  # 线程， batch
        return 1, 32

    def get_feature_len(self) -> int:
        logger.info('in get_feature_len')
        return self.feature_len

    def get_feature_multiprocess(self, images: List[ndarray]) -> Tuple[List[bool], List[ndarray]]:
        """多线程实现 get_feature 函数功能"""
        is_s = []
        feats = []

        # 定义一个内部函数来处理每个图像，并能捕获异常
        def process_image(image: ndarray) -> Tuple[bool, ndarray]:
            try:
                face_image = self._get_faces(image)
                if not isinstance(face_image, list):
                    face_image = [face_image]
                input_size = self.model.get_inputs()[0].shape[-2:]
                blob = cv2.dnn.blobFromImages(face_image, 1.0 / 127.5, input_size,
                                              (127.5, 127.5, 127.5), swapRB=True, crop=False)
                net_out = self.model.run(None, {self.model.get_inputs()[0].name: blob})[0]
                return True, net_out
            except Exception as e:
                # 可以记录异常信息以便调试
                print(f"Error processing image: {e}")
                return False, None

        # 使用线程池来并行处理每个图像
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(process_image, image) for image in images]

            # 收集结果
            for future in futures:
                result = future.result()
                is_s.append(result[0])
                feats.append(result[1])

        return is_s, feats

    def get_feature(self, images: List[ndarray]) -> Tuple[List[bool], List[ndarray]]:
        is_s = []
        face_image_list = []
        for image in images:
            isS = True
            try:
                face_image = self._get_faces(image)
                if face_image is None:
                    face_image = np.zeros((112, 112, 3), dtype=np.uint8)
                    isS = False
            except Exception:
                logger.error('ERROR: det error')
                isS = False
                traceback.print_exc()
                face_image = cv2.resize(image, (112, 112))
            face_image_list.append(face_image)
            is_s.append(isS)

        blob = cv2.dnn.blobFromImages(
            face_image_list,
            1.0 / 127.5,
            [112, 112],
            (127.5, 127.5, 127.5),
            swapRB=True, crop=False
        )
        net_out = self.model.run(None, {self.model_input_name: blob})[0]
        feat_list = [i for i in net_out]
        # for i, usable in enumerate(is_s):
        #     if not usable:
        #         feat_list[i] = np.zeros((512,))
        # self.gallery_usable_list += is_s
        return is_s, feat_list

    def insert_gallery(self, feat: ndarray, idx: int, label: int, usable=True) -> None:
        self.gallery_list.append(feat)
        self.gallery_usable_dict[idx] = usable
        # feat = feat[np.newaxis, :]
        # faiss.normalize_L2(feat)
        # self.gallery.add(feat)
        self.idx_map.append(idx)

    def finalize(self) -> None:
        np_gallery = np.array(self.gallery_list)
        faiss.normalize_L2(np_gallery)
        is_usable_list = []
        for idx_map_item in self.idx_map:
            is_usable_list.append(self.gallery_usable_dict[idx_map_item])
        np_gallery[~np.array(is_usable_list), :] = 0
        index = faiss.IndexFlatIP(512)
        res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        self.gpu_index.add(np_gallery)

    def get_topk(self, query_feats: List[ndarray], usable: List[bool]) -> Tuple[List[ndarray], List[ndarray]]:
        query_feats = np.array(query_feats)
        faiss.normalize_L2(query_feats)
        # 将某些特征置零
        query_feats[~np.array(usable), :] = 0
        distances_old, indices = self.gpu_index.search(query_feats, 11)
        indices = [[self.idx_map[i[0]]] for i in indices]
        #distances = [i for i in distances]
        #return indices, distances
        distances = []
        for dist in distances_old:
            adjusted_distance = dist[0] - np.mean(dist[1:])
            distances.append([adjusted_distance])
        return indices, distances
        # TODO 找到最相近的11张人脸，距离用第一张的距离减去后十张距离的均值（尝试提升分数），已解决

    def feature_to_str(self, query_feats: List[ndarray]) -> List[dict]:

        feature_zq = [{"feature": ['TZSJ'], "quality": 'str'}]
        return feature_zq

    #def get_sim(self, feat1: ndarray, feat2: ndarray) -> float:
        #feat1 = feat1.flatten()
        #feat2 = feat2.flatten()
        #return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

    def _get_faces(self, image):

        max_num = 0
        metric = 'default'
        bboxes_list = []
        scores_list = []
        kpss_list = []

        input_size = (320, 320)
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        input_size = tuple(det_img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.detect_model.run(self.output_names, {self.detect_model.get_inputs()[0].name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= 0.5)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)

        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        if len(det) < 1:
            return None
        """从检测到的多张脸中选择一张脸 """
        # TODO det中可能一张人脸也没有，此时应该将特征直接置0，确定kpss不为空，否则建模失败(尝试解决建模失败问题）,已解决
        best_idx = select_face(det=det, img_shape=image.shape)
        img = norm_crop(image, kpss[best_idx])
        return img

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


if __name__ == '__main__':
    import time
    images_diku = {i.stem: cv2.imread(i) for i in sorted(Path('images/gallery').glob('*.jpg'))}
    images_test = {i.stem: cv2.imread(i) for i in sorted(Path('images/probe').glob('*.jpg'))}

    fat = PyFAT(6, 1)
    fat.load()
    time_pre = time.time()
    k_list, v_list = [], []
    for k, v in images_diku.items():
        k_list.append(k)
        v_list.append(v)

    is_s, feats = fat.get_feature(v_list)
    for idx, k in enumerate(k_list):
        usable = is_s[idx]
        fat.insert_gallery(feats[idx], idx=int(k), label=0, usable=usable)

    # 模拟平台上的报错
    is_s, feats = fat.get_feature(v_list)
    for idx, k in enumerate(k_list):
        usable = is_s[idx]
        fat.insert_gallery(feats[idx], idx=int(k), label=0, usable=usable)

    fat.finalize()

    k_list, v_list = [], []
    for k, v in images_test.items():
        k_list.append(k)
        v_list.append(v)
    is_s, feats = fat.get_feature(v_list)
    idxs, sims = fat.get_topk(feats, is_s)
    for index, k in enumerate(k_list):
        print(f'探测: {k}, 检索到底库图片idx: {idxs[index]}, 相似度: {sims[index][0]}')
