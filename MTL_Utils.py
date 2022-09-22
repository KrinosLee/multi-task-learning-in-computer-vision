import torch
import numpy as np

def Loss_DIoU(box_pre, box_true, device):
    
    # DIoU loss function: https://arxiv.org/abs/1911.08287
    # based on GIoU and plus a penalty items(central point distance)
    # DIoU = 1 - IoU + normal_dis (normal_dis = center_diagonal^2 / outer_diagonal^2)
    # IoU = Area_overlap / Area_all
    
    # compute the area of each box
    pre_area = (box_pre[:,2]-box_pre[:,0])*(box_pre[:,3]-box_pre[:,1])
    true_area = (box_true[:, 2] - box_true[:, 0]) * (box_true[:, 3] - box_true[:, 1])

    # compute the overlapped area
    over_left_up = torch.max(box_pre[...,:2], box_true[...,:2])
    over_right_down = torch.min(box_pre[...,2:], box_true[...,2:])
    over_boundary = torch.max(over_right_down-over_left_up, torch.tensor([0,0], device=device))
    area_overlap = over_boundary[:,0] * over_boundary[:,1]
    area_all = pre_area + true_area - area_overlap
    
    # compute IoU
    iou = area_overlap/area_all
    
    # compute the outer box diagonal
    outer_left_up = torch.min(box_pre[:,:2], box_true[:,:2])
    outer_right_down = torch.max(box_pre[:,2:], box_true[:,2:])
    outer_boundary = torch.max(outer_right_down - outer_left_up, torch.tensor([0,0], device=device))
    outer_diagonal = outer_boundary[:,0]**2 + outer_boundary[:,1]**2

    # compute the center distance
    box_pre_center = (box_pre[:,:2] + box_pre[:,2:]) / 2
    box_true_center = (box_true[:,:2] + box_true[:,2:]) / 2
    center_distance = (box_pre_center[:,0]-box_true_center[:,0])**2 \
        + (box_pre_center[:,1]-box_true_center[:,1])**2

    # compute DIoU
    diou = 1 - iou + center_distance / outer_diagonal

    return diou.mean()

def Metrics_IoU(box_pre, box_true, device):
    box_pre, box_true = box_pre.detach(), box_true.detach()
    # IoU = Area_overlap / Area_all
    
    # compute the area of each box
    pre_area = (box_pre[:,2]-box_pre[:,0])*(box_pre[:,3]-box_pre[:,1])
    true_area = (box_true[:, 2] - box_true[:, 0]) * (box_true[:, 3] - box_true[:, 1])

    # compute the overlapped area
    over_left_up = torch.max(box_pre[...,:2], box_true[...,:2])
    over_right_down = torch.min(box_pre[...,2:], box_true[...,2:])
    over_boundary = torch.max(over_right_down-over_left_up, torch.tensor([0,0], device=device))
    area_overlap = over_boundary[:,0] * over_boundary[:,1]
    area_all = pre_area + true_area - area_overlap
    
    # compute IoU
    iou = area_overlap/area_all

    return iou.mean()

def Weight_DWA(loss_his, num_task, pre_weight, temper=2):
    if num_task != len(pre_weight): raise Exception("num_task do not match length of pre_weight")
    pre_weight = torch.tensor(pre_weight)
    if loss_his == None: return pre_weight

    loss_his = loss_his.T
    w = loss_his[1] / loss_his[0]
    division = 0
    for i in range(num_task):
        division += torch.exp(w[i] / temper)
    dwa = torch.exp(w / temper) / division
    # print(dwa,dwa.sum())
    pre_w_dwa = pre_weight * dwa
    # print(pre_w_dwa, pre_w_dwa.sum())

    weight = pre_w_dwa
    return num_task * weight / weight.sum()

def Class_Acc(y_pre, y_true):
    y_pre, y_true = y_pre.detach(), y_true.detach()
    length = len(y_pre)
    num_correct = torch.eq(y_pre, y_true).sum().float().item()
    return num_correct / length

def per_class_iu(hist):
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


def Seg_mIoU(num_class, pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    batch_len = len(pred)
    ious_per_img = np.zeros(batch_len)
    index=0
    for a, b in zip(pred, label):
        a = a.reshape(-1)
        b = b.reshape(-1)
        n=num_class
        k = (a >= 0) & (a < num_class)
        cfsmatrix = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
        ious = dict(zip(range(n), per_class_iu(cfsmatrix)))
        total_iou = 0
        count = 0
        for key, value in ious.items():
            if isinstance(None, list) and key in None or isinstance(None, int) and key == None:
                continue
            total_iou += value
            count += 1
        ious_per_img[index] = total_iou / count
        index += 1
    return np.mean(ious_per_img)

def Seg_IoU(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    iou = []
    for pred_i, label_i in zip(pred, label):
        and_v = np.sum(pred_i & label_i)
        or_v = np.sum(pred_i | label_i)
        iou.append(and_v / or_v)
    return np.mean(iou)
    
