import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
from models.experimental import attempt_load
from utils.general import non_max_suppression

weights = './yolov5n.pt'

imgsz = 640
conf_thres = 0.25
max_det = 1000
hide_conf = True

frame_rate_calc = 1
freq = cv2.getTickFrequency()

color_blue = (255, 255, 0)
color_red = (25, 20, 240)
color = color_blue
text_x_align = 10
inference_time_y = 30
fps_y = 90
analysis_time_y = 60
font_scale = 0.7
thickness = 2
rect_thickness = 3

pred_shape = (480, 640, 3)
vis_shape = (800, 600)

cap = cv2.VideoCapture(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# half &= device.type != 'cpu'

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

while 1:
    t1 = cv2.getTickCount()
    start = datetime.now()

    ret, frame = cap.read()
    out = frame.copy()

    frame = cv2.resize(frame, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_LINEAR)
    frame = np.transpose(frame, (2, 1, 0))

    cudnn.benchmark = True  # set True to speed up constant image size inference

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    frame = torch.from_numpy(frame).to(device)
    frame = frame.float()
    frame /= 255.0
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)

    frame = torch.transpose(frame, 2, 3)

    s = datetime.now()
    pred = model(frame, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, max_det=max_det)
    e = datetime.now()
    d = e - s
    inf_time = round(d.total_seconds(), 3)

    # detections per image
    for i, det in enumerate(pred):

        img_shape = frame.shape[2:]
        out_shape = out.shape

        s_ = f'{i}: '
        s_ += '%gx%g ' % img_shape  # print string

        if len(det):

            coords = det[:, :4]

            gain = min(img_shape[0] / out_shape[0], img_shape[1] / out_shape[1])  # gain  = old / new
            pad = (img_shape[1] - out_shape[1] * gain) / 2, (
                    img_shape[0] - out_shape[0] * gain) / 2  # wh padding

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain

            coords[:, 0].clamp_(0, out_shape[1])  # x1
            coords[:, 1].clamp_(0, out_shape[0])  # y1
            coords[:, 2].clamp_(0, out_shape[1])  # x2
            coords[:, 3].clamp_(0, out_shape[0])  # y2

            det[:, :4] = coords.round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s_ += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):

                c = int(cls)  # integer class
                label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'

                tl = rect_thickness

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(out, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(out, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(out, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)

    end = datetime.now()
    duration = end - start
    a_time = round(duration.total_seconds(), 3)

    inference_time = 'Inference Time: {}'.format(inf_time)
    label_size, base_line = cv2.getTextSize(inference_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            thickness)
    label_ymin = max(inference_time_y, label_size[1] + 10)
    cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                  (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                  cv2.FILLED)
    cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                  (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
    cv2.putText(out, inference_time, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color_blue,
                thickness,
                cv2.LINE_AA)

    fps = 'FPS: {0:.2f}'.format(frame_rate_calc)
    label_size, base_line = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    label_ymin = max(fps_y, label_size[1] + 10)
    cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                  (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                  cv2.FILLED)
    cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                  (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
    cv2.putText(out, fps, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color_blue,
                thickness,
                cv2.LINE_AA)

    analysis_time = 'Analysis Time: {}'.format(a_time)
    label_size, base_line = cv2.getTextSize(analysis_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            thickness)
    label_ymin = max(analysis_time_y, label_size[1] + 10)
    cv2.rectangle(out, (text_x_align, label_ymin - label_size[1] - 10),
                  (text_x_align + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                  cv2.FILLED)
    cv2.rectangle(out, (text_x_align - 2, label_ymin - label_size[1] - 12),
                  (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
    cv2.putText(out, analysis_time, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color_blue,
                thickness,
                cv2.LINE_AA)
    out = cv2.resize(out, vis_shape, cv2.INTER_LINEAR)
    cv2.imshow("out", out)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(5) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break