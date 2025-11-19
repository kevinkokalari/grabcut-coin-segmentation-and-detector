import cv2 as cv, numpy as np, tensorflow as tf
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
import cvzone
import json, numpy as np
#from debug import Debug
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




# ---------- load CNN once ----------
MODEL  = tf.keras.models.load_model("model/yen_cnn.h5")
IMG_SZ = MODEL.input_shape[1]
VALUES = {"1":1,"5":5,"10":10,"50":50,"100":100,"500":500}

with open("model/labels.json") as f:        
    idx_map = json.load(f)                  

LABELS = [lbl for lbl, _ in sorted(idx_map.items(),
                                   key=lambda kv: kv[1])]


def load_true_labels(path):
    ret_array = [""]
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            obj = json.loads(line)
            print(int(lineno))
            ret_array.append(list(obj.values())[0])
    print_dist = {"1":0,"5":0,"10":0,"50":0,"100":0,"500":0}
    for tr in ret_array[1:]:
        print_dist["1"] += tr["1"]
        print_dist["5"] += tr["5"]
        print_dist["10"] += tr["10"]
        print_dist["50"] += tr["50"]
        print_dist["100"] += tr["100"]
        print_dist["500"] += tr["500"]

    print("----------------------------")
    print("Test-set Class Distribution:")
    print("1¥:   ", print_dist["1"])
    print("5¥:   ", print_dist["5"])
    print("10¥:  ", print_dist["10"])
    print("50¥:  ", print_dist["50"])
    print("100¥: ", print_dist["100"])
    print("500¥: ", print_dist["500"])
    print("----------------------------")
    input("Press any key to continue...")
    return ret_array



def _enlarge_box(x, y, w, h, scale, W, H):
    """
    Enlarges the found box with coin with a given scaling factor
    """
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w * scale, h * scale
    x_new  = int(max(0, cx - w2 / 2))
    y_new  = int(max(0, cy - h2 / 2))
    w_new  = int(min(W, cx + w2 / 2) - x_new)
    h_new  = int(min(H, cy + h2 / 2) - y_new)
    return x_new, y_new, w_new, h_new


def preprocess_roi(src, bbox):
    x,y,w,h = bbox
    pad = int(0.05 * max(w, h))                    # 5% extra padding
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, src.shape[1])
    y1 = min(y + h + pad, src.shape[0])
    roi = src[int(y0):int(y1), int(x0):int(x1)]
   
    yuv = cv.cvtColor(roi, cv.COLOR_BGR2YUV)
    y_chan, u, v = cv.split(yuv)
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(3,3))
    y_eq  = clahe.apply(y_chan) #CLAHE on Y-channel
    roi   = cv.cvtColor(cv.merge((y_eq,u,v)), cv.COLOR_YUV2BGR)
    #cv.imshow("Preprocess ROI",roi); cv.waitKey(0)
    roi = cv.resize(roi, (IMG_SZ, IMG_SZ))
    return roi


def cnn_classify(roi_bgr, thr=0.4):
    #cv.imshow("input CNN", roi_bgr); cv.waitKey(0)
    roi = roi_bgr[..., ::-1] / 255.0               # BGR→RGB, scaled to [0,1]
    pred = MODEL.predict(roi[np.newaxis], verbose=0)[0]

    idx  = int(np.argmax(pred));  
    conf = float(pred[idx])
    print("-----------------------------------")
    print(conf)
    print("-----------------------------------")
    if conf < thr:
        return [], 0
    return LABELS[idx], conf


def find_coins(img_bgr, visualize=False):
    H, W = img_bgr.shape[:2]

    # --- detect circles ------------------------------------------
    #gray  = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)


    gray = img_bgr[:,:,1]

    clahe = cv.createCLAHE(clipLimit=1.7, tileGridSize=(8,8))
    gray  = clahe.apply(gray)


    #img_bgr= cv.cvtColor(cv.merge((y_chan,u,v)), cv.COLOR_YUV2BGR)

    #gray = img_bgr[:,:,1]

    #gray = cv.equalizeHist(gray)


    gray  = cv.GaussianBlur(gray, (11, 11), 2.2)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=2.2, minDist=10,
                           param1=200, param2=100, minRadius=70, maxRadius=205)

    # --- build seed mask -----------------------------------------
    gc_mask = np.full((H, W), cv.GC_BGD, np.uint8)   # default bg

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            cv.circle(gc_mask, (x, y), int(1*r), cv.GC_FGD,    -1)  # sure FG
            cv.circle(gc_mask, (x, y), int(1.3*r), cv.GC_PR_FGD, -1)  # probable

    else:                               # if Hough fails, treat area around middle point of img as FG
        rect = (int(.25*W), int(.25*H), int(.75*W), int(.75*H))
        cv.grabCut(img_bgr, gc_mask, rect,
                   np.zeros((1, 65), np.float64),
                   np.zeros((1, 65), np.float64),
                   iterCount=1, mode=cv.GC_INIT_WITH_RECT)

    # --- single GrabCut iteration --------------------------------
    cv.grabCut(img_bgr, gc_mask, None,
               np.zeros((1, 65), np.float64),
               np.zeros((1, 65), np.float64),
               iterCount=1, mode=cv.GC_INIT_WITH_MASK)

    fg = (gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD)
    #fg = cv.morphologyEx(fg.astype(np.uint8), cv.MORPH_OPEN,
    #                 np.ones((3,3), np.uint8), iterations=1).astype(bool)

    # --- connected components ------------------------------------
    num, labels, stats, _ = cv.connectedComponentsWithStats(fg.astype(np.uint8))
    out = []
    scale = 1.1
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if 0.0004*H*W < a < 0.25*H*W:   # drop lint / huge blobs
            x, y, w, h = _enlarge_box(x, y, w, h, scale, W, H)
            out.append(((x, y, w, h), labels == i))

    # --- visualisation -------------------------------------------
    if visualize:
        dbg = img_bgr.copy()
        dbg[fg] = cv.addWeighted(
            img_bgr, 0.3, np.full_like(img_bgr, (0, 255, 0)), 0.7, 0
        )[fg]
        for (x, y, w, h), _ in out:
            cv.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow("GrabCut seeded by circles", dbg)
        cv.waitKey(0); cv.destroyAllWindows()

    return out

def det_rate_per_img(pred, true):
    print("pred: ", pred)
    print("true: ", true)


    tp = fp = fn = 0
    for denom in pred.keys() | true.keys():     # union of all coin types
        p = pred.get(denom, 0)
        t = true.get(denom, 0)

        tp += min(p, t)             # correct coins
        fp += max(0, p - t)         # extra coins
        fn += max(0, t - p)         # missed coins

    total = tp + fp + fn
    return 1.0 if total == 0 else tp / total



def run_on_image(path, filename):
    img = cv.imread(str(path)); tot = 0
    H, W = img.shape[:2]
    labels = {'0': 0, '1': 0, '5': 0, '10': 0, '50': 0, '100': 0, '500': 0}



    for (x,y,w,h), _ in find_coins(img, DEBUG):
        diag       = (w**2 + h**2) ** 0.5      # coin “size”
        font_scale = max(2, diag / 200)      
        thickness  = int(font_scale * 2)
        
        roi = preprocess_roi(img, (x, y, w, h))
        #cv.imshow("test", roi); cv.waitKey(0)
        label, conf = cnn_classify(roi)
        if label == []:
            continue
        val = VALUES[label];  
        tot += val

        labels[label] += 1

        # --------- cast to int before drawing --------------
        x, y, w, h = map(int, (x, y, w, h))
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cvzone.putTextRect(img, f"{val} JPY", (x, y - 5), scale=font_scale, thickness=thickness)
        print(f"{val} JPY")

    font_scale = min(6, 0.005 * H)        # grows linearly with resolution
    thickness  = int(font_scale * 2)
    det = det_rate_per_img(labels, true[int(filename.split(".")[0])])
    print("=================================================================")
    print("Total JPY: ", tot)
    print("Detection Rate for image: ", det) 
    print("-----------------------------------------------------------------")
    print("Select the 'Yen CNN' window and then press any key to continue...")      
    print("=================================================================")
    cvzone.putTextRect(img,f"Total {tot} JPY",(40,80),scale=font_scale, thickness=thickness)
    cv.imshow("Yen CNN",img); cv.waitKey(0)

    return det

CLASSES = ["1", "5", "10", "50", "100", "500"]
DUMMY   = "None"                 # to catch FP and FN


def image_to_counts(img_path: Path) -> dict[str, int]:
    """
    Run pipeline on ONE image and return the per-class counts dict.
    """
    img   = cv.imread(str(img_path))
    counts = {c: 0 for c in CLASSES}

    for (x, y, w, h), _ in find_coins(img, DEBUG):
        roi   = preprocess_roi(img, (x, y, w, h))
        label, _ = cnn_classify(roi)
        if label:                        # skip empty result
            counts[label] += 1
    return counts


def build_confusion_matrix(image_dir: Path, gt_dict: dict[int, dict[str, int]]) -> None:
    """
    Produces a plotted confusion matrix.
    """
    y_true, y_pred = [], []

    img_files = [
    p for p in image_dir.rglob("*")               # recurse into sub-dirs
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    print("Images found:", len(img_files))
    for img_file in sorted(img_files):   
        img_id   = int(img_file.stem)                  # "17.jpg" → 17
        try:
            true_cnt = gt_dict[img_id]
            pred_cnt = image_to_counts(img_file)
        except Exception as e:
            print(e)
            print(img_id)
            input()

        # expand counts into 1D label lists
        true_list = [c for cls, n in true_cnt.items()  for c in [cls]*n]
        pred_list = [c for cls, n in pred_cnt.items()  for c in [cls]*n]

        # pad with DUMMY so both lists are equal length
        if len(true_list) > len(pred_list):
            pred_list.extend([DUMMY] * (len(true_list) - len(pred_list)))
        elif len(pred_list) > len(true_list):
            true_list.extend([DUMMY] * (len(pred_list) - len(true_list)))

        y_true.extend(true_list)
        y_pred.extend(pred_list)

    labels = CLASSES + [DUMMY]
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap="Blues")

    ax = plt.gca()                     # axis returned by .plot()
    ax.xaxis.tick_top()                # tick marks / tick labels
    ax.xaxis.set_label_position('top') # the axis label
    ax.set_xlabel("Predicted label", labelpad=10) 

    acc = np.trace(cm) / cm.sum()            # diagonal / total
    ax.text(0.95, -0.10,                     # (x, y) in axis fraction coords
        f"Accuracy coin prediction = {acc:.2%}",
        transform=ax.transAxes,
        ha="right", va="center", fontsize=12)


    plt.title("Coin classification confusion matrix")
    plt.tight_layout()
    plt.show()

    

def load_label_array(path: str | Path) -> dict[int, dict[str, int]]:
    mapping = {}
    with Path(path).open() as f:
        for line in f:
            obj = json.loads(line)
            img_id, counts = next(iter(obj.items()))
            mapping[int(img_id)] = counts
    return mapping




if __name__=="__main__":

    CREATE_CONF_MATRIX = False
    DEBUG = False

    total_acc = []
    path = Path("./test")
    true = load_true_labels("./" + str(path)+ "/true.jsonl")
    if not CREATE_CONF_MATRIX:
        for file in path.iterdir(): 
            if file.is_file() and file.name != ".DS_Store" and file.name !="true.jsonl":
                filename = "./" + str(path)+ "/" + file.name
                total_acc.append(run_on_image(filename, file.name))
            

        print("Avg Detection Rate per Image: ", sum(total_acc)/len(total_acc))
    else:
        build_confusion_matrix(path, true)
    