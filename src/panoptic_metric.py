# mIoU
intersection = np.zeros((len(t_color)-1))
union = np.zeros((len(t_color)-1))
iou = np.zeros((len(t_color)-1))
t_area = np.zeros((len(t_color)-1))

for ind in tqdm(range(len(img_path))):
    img = np.asarray(Image.open(img_path[ind]))
    img_t = np.asarray(Image.open(img_t_path[ind]))

    img = preprocess(img)

    inputs = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))
    inputs[0] = transform(img)

    with torch.no_grad():
        outputs = model(inputs.to(device))

    output = outputs[4].cpu().detach().numpy()
    output = np.array(np.argmax(output[0], axis=0))

    for cls_ in range(len(t_color)-1):
        output_cls = np.where(output==cls_, 1, 0)
        img_t_cls = np.where((img_t[:, :, 0] == t_color[cls_][0])
                             & (img_t[:, :, 1] == t_color[cls_][1])
                             & (img_t[:, :, 2] == t_color[cls_][2]), 1, 0)
        union[cls_] += np.sum(np.where((output_cls + img_t_cls)>0, 1, 0))
        intersection[cls_] += np.sum(output_cls * img_t_cls)
        t_area[cls_] += np.sum(img_t_cls)

for cls_ in range(len(t_color)-1):
    if union[cls_] == 0:
        iou[cls_] = 0
    else:
        iou[cls_] = intersection[cls_] / union[cls_]

with open(metrics + "metrics_mIoU", mode='w') as f:
    f.write("mIoU:{:.3f}\n"
            .format(np.mean(np.array(iou))))
    for i in range(len(t_color)-1):
        f.write("{}, IoU:{:.3f}, {}, union:{}, t_area:{}, intersection:{}\n"
            .format(i, iou[i], t_class_name[i], union[i],
                    t_area[i], intersection[i]))

np.save(metrics + 'intersection', intersection)
np.save(metrics + 'union', union)
np.save(metrics + 'iou', iou)
np.save(metrics + 't_area', t_area)

# PQ(Panoptic Quality), AP(Average Precision)の計算
TP = np.zeros((len(t_color)-1))
FP = np.zeros((len(t_color)-1))
FN = np.zeros((len(t_color)-1))
IoU_05 = [[] for i in range(len(t_color)-1)]
IoU = np.zeros((len(t_color)-1))
AP = np.zeros((len(t_color)-1))
SQ = np.zeros((len(t_color)-1))
RQ = np.zeros((len(t_color)-1))

st_for=0
en_for=4
min_size=20

for ind in tqdm(range(len(img_path))):
    img = np.asarray(Image.open(img_path[ind]))
    seg_t = np.asarray(Image.open(img_t_path[ind]))
    ins_t = np.asarray(Image.open(img_ins_t_path[ind]))

    img = preprocess(img)
    
    inputs = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))

    inputs[0] = transform(img)
    
    with torch.no_grad():
        outputs = model(inputs.to(device))
        
    detect, ins_list = make_ins_seg(outputs, st_for=st_for,
                                    en_for=en_for, min_size=min_size)

    for cls_ in range(len(t_color)-1):
        # Falseだと、そのクラスが画像中に存在していないことを示す.
        t_flag = False
        d_flag = False

        # 正解ラベル
        # 該当クラスのinstanceのみ取り出す
        mask = np.where((seg_t[:, :, 0] == t_color[cls_][0])
                        & (seg_t[:, :, 1] == t_color[cls_][1])
                        & (seg_t[:, :, 2] == t_color[cls_][2]),
                        1, 0)[..., None]
        ins_t_cls = ins_t * mask
        if np.max(ins_t_cls)!=0:
            t_flag = True
            # instanceを一つずつ224x224のndarrayに分ける
            ins_colors = ins_t_cls.reshape(IMG_SIZE*IMG_SIZE, 3)
            ins_colors = sorted(list(map(list, set(map(tuple, ins_colors)))))[::-1]
            if [0, 0, 0] in ins_colors:
                ins_colors.remove([0, 0, 0])
            t_com = np.zeros((len(ins_colors), IMG_SIZE, IMG_SIZE))

            for i in range(len(ins_colors)):
                t_com[i] = np.where((ins_t_cls[:, :, 0] == ins_colors[i][0])
                                    & (ins_t_cls[:, :, 1] == ins_colors[i][1])
                                    & (ins_t_cls[:, :, 2] == ins_colors[i][2]), 1, 0)

        # 識別結果
        # instanceを一つずつ224x224のndarrayに分ける
        if np.max(detect[cls_])!=0:
            d_flag = True
            ins_colors = detect[cls_].reshape(IMG_SIZE*IMG_SIZE, 3)
            ins_colors = sorted(list(map(list, set(map(tuple, ins_colors)))))[::-1]
            if [0, 0, 0] in ins_colors:
                ins_colors.remove([0, 0, 0])
            det_com = np.zeros((len(ins_colors), IMG_SIZE, IMG_SIZE))
            for i in range(len(ins_colors)):
                det_com[i] = np.where((detect[cls_][:, :, 0] == ins_colors[i][0])
                                      & (detect[cls_][:, :, 1] == ins_colors[i][1])
                                      & (detect[cls_][:, :, 2] == ins_colors[i][2]), 1, 0)

        # t_comとdet_comを比較.
        if t_flag == False and d_flag == False:
            continue
        elif t_flag == True and d_flag == False:
            FN[cls_] += len(t_com)
        elif t_flag == False and d_flag == True:
            FP[cls_] += len(det_com)
        else:
            TP_cls = 0
            for i in range(len(det_com)):
                union = np.sum(np.sum(np.clip(t_com + det_com[i], 0, 1), axis=2), axis=1)
                intersection = np.sum(np.sum(t_com * det_com[i], axis=2), axis=1)

                max_IoU = np.max(intersection / union)
                if max_IoU > 0.5:
                    TP_cls += 1
                    IoU[cls_] += max_IoU
                    IoU_05[cls_].append(max_IoU)
            TP[cls_] += TP_cls
            FP[cls_] += len(det_com) - TP_cls
            FN[cls_] += len(t_com) - TP_cls
            
    if ind % 100 == 0:
        with open(metrics + "log", mode='a') as f:
            f.write("index:{}\n".format(ind))

np.save(metrics + 'TP', TP)
np.save(metrics + 'FP', FP)
np.save(metrics + 'FN', FN)
np.save(metrics + 'PQ_IoU', IoU)

for cls_ in range(len(t_color)-1):
    # Segmentation Quality
    if TP[cls_] == 0:
        SQ[cls_] = 0
    else:
        SQ[cls_] = np.array(IoU[cls_]) / np.array(TP[cls_])
    # Recognition Quality
    denominator = TP[cls_] + FP[cls_]/2 + FN[cls_]/2
    if denominator == 0:
        RQ[cls_] = 0
    else:
        RQ[cls_] = TP[cls_] / denominator
    
    # APの計算
    TP_FP = TP[cls_] + FP[cls_]
    if TP_FP == 0:
        AP[cls_] = 0
        continue
    for thresh in range(50, 100, 5):
        th = thresh / 100
        iou_05 = np.asarray(IoU_05[cls_])
        AP[cls_] += np.sum(np.where(iou_05>th, 1, 0)) / TP_FP
AP = AP / 10

# Panoptic Quality
PQ = SQ * RQ

with open(metrics + "metrics_mAP_mPQ", mode='w') as f:
    f.write("mAP{}, mPQ:{}\n"
            .format(np.mean(AP), np.mean(PQ)))
    for i in range(len(t_color)-1):
        f.write("{}, AP:{:.3f}, PQ:{:.3f}, {}\n"
            .format(i, AP[i], PQ[i], t_class_name[i]))

np.save(metrics + 'SQ', SQ)
np.save(metrics + 'RQ', RQ)
np.save(metrics + 'PQ', PQ)
np.save(metrics + 'AP', AP)