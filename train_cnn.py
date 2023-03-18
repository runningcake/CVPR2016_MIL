import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import os
import numpy as np
from utils.network import model_resnet50_as_backbone
from utils.dataset import MyDataset


NUM_CLASSES = 3 # ADC, SCC, ADC_mix
PATCHES_DIR = "./patches" # the path of your patches
model = model_resnet50_as_backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


all_paths = [os.path.join(PATCHES_DIR, x) for x in os.listdir(PATCHES_DIR)]
ids = []
for path in all_paths:
    id_ = "_".join(path.split("/")[-1].split("_")[:2])
    if id_ not in ids:
        ids.append(id_)

lr = 0.001
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.1*lr,
    eps=1e-4,
    amsgrad=True
)

def np_gen_gaussian_kernel(w, sigma=1, normalized=False):
    assert type(w) == int
    kernel = np.array([[
        np.exp(-((i-w)**2+(j-w)**2) / (2*sigma**2)) for j in range(2*w + 1)
    ] for i in range(2*w + 1)
    ])
    if normalized:
        return kernel / kernel.sum()
    return kernel

def get_smoothed_patch_prob(path, w=4):
    kernel = np_gen_gaussian_kernel(w)
    raw_np = np.zeros((2*w+1, 2*w+1))
    id_ = "_".join(path.split("/")[-1].split("_")[:2])
    i_p, j_p = (int(x) for x in path[:-4].split("_")[-2:])
    for delta_i in range(-w, w+1):
        for delta_j in range(-w, w+1):
            i_k = i_p + delta_i
            j_k = j_p + delta_j
            raw_path = os.path.join(PATCHES_DIR, f"{id_}_{i_k}_{j_k}.jpg")
            # in all_paths<list> is O(n) in time, while in <dict> is O(1).
            if raw_path in probs_by_patch:
                raw_np[delta_i+w, delta_j+w] = probs_by_patch[raw_path]
            else:
                kernel[delta_i+w, delta_j+w] = 0
                raw_np[delta_i+w, delta_j+w] = 0
    kernel /= kernel.sum()
    return (kernel * raw_np).sum()

iteration = 0
discriminative_patches = all_paths
model.to(device)
try:
    np_records = [file for file in os.listdir("./record") \
                  if file.startswith("res_record_")]
    np_records = sorted(np_records, key=lambda x: os.path.getmtime(
        os.path.join("./record", x)
    ))
    record = np_records[-1] # latest record
    rec = np.load(os.path.join("./record", record), allow_pickle=True).item()
    iteration = rec["iteration"]
    discriminative_patches = rec["discriminative_patches"]
    
    state_dict = torch.load("./record/res_iteration.pth", map_location=device)
    model.load_state_dict(state_dict["model_state"])
    optimizer.load_state_dict(state_dict["optimizer_state"])
    
    print(f"model loaded. iteration: {iteration}")
    print(f"discriminative patches: {len(discriminative_patches)}")
    iteration += 1
except:
    iteration = 0
    discriminative_patches = all_paths

while True:
    iter_dst = MyDataset(paths=discriminative_patches)
    criterion = nn.CrossEntropyLoss(weight=iter_dst.weights.to(device))
    loader = data.DataLoader(
        iter_dst, batch_size=64, shuffle=True, num_workers=12,
        drop_last=False)
    old_discriminative_patches = discriminative_patches
    discriminative_patches = []
    
    # M-step, train 2 epochs on discriminative patches
    for epoch in range(2):
        inner_itrs = 0; interval_loss = 0
        for images, labels, paths in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).view(-1, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_iter = loss.detach().cpu().item()
            interval_loss += loss_iter
            
            n_calc_loss = 100
            if (inner_itrs > 0) & ((inner_itrs) % n_calc_loss == 0):
                interval_loss = interval_loss / n_calc_loss
                print("Iteration %d, Epoch %d, Inner_Itrs %d, Loss=%f" %
                      (iteration, epoch, inner_itrs, interval_loss))
                iterval_loss = 0
            
            inner_itrs += 1
        
    # E-step, predict all probs, smoothing and thresholding
    
    probs_by_patch = {}; smoothed_by_patch = {}
    # thresholds for class
    probs_by_class = {}
    probs_by_class[0] = []; probs_by_class[1] = []; probs_by_class[2] = []
    # thresholds for id
    probs_by_id = {}; dic_id_threshold = {}
    
    # 先完成所有 patch
    # 再算出所有 smoothed
    # 最后算 threshold
    dst = MyDataset(paths=all_paths)
    loader = data.DataLoader(
        dst, batch_size=64, shuffle=True, num_workers=12,
        drop_last=False)
    model.eval()
    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            outputs = model(images).view(-1, 3)
            probs = F.softmax(outputs.detach().cpu().float(), dim=1)
            for j, path in enumerate(paths):
                id_ = "_".join(path.split("/")[-1].split("_")[:2])
                # e.g. class0_2f9acd02-6e70-482f-a1fb-9f72a3cdcc28
                class_ = int(id_[5])
                prob = probs[j, class_].item()
                probs_by_patch[path] = prob
    for path in all_paths:
        smoothed_prob = get_smoothed_patch_prob(path)
        smoothed_by_patch[path] = smoothed_prob
        id_ = "_".join(path.split("/")[-1].split("_")[:2])
        class_ = int(id_[5])
        if id_ not in probs_by_id:
            probs_by_id[id_] = []
        probs_by_id[id_].append(smoothed_prob)
        probs_by_class[class_].append(smoothed_prob)

    # find quantiles
    thresh_by_class = {k: x for k, x in enumerate(list(map(
        lambda x: np.quantile(x, 0.28),
        [probs_by_class[class_] for class_ in range(3)]
    )))}
    thresh_by_id = {k: np.quantile(v, 0.25) for (k, v) in probs_by_id.items()}

    # find discriminative patches
    for path in all_paths:
        id_ = "_".join(path.split("/")[-1].split("_")[:2])
        class_ = int(id_[5])
        threshold = min(thresh_by_id[id_], thresh_by_class[class_])
        if smoothed_by_patch[path] >= threshold:
            discriminative_patches.append(path)
    n_same = len(set(old_discriminative_patches) &\
                 set(discriminative_patches))
    record_dic = {
        "iteration": iteration,
        "old_discriminative_patches": tuple(old_discriminative_patches),
        "discriminative_patches": tuple(discriminative_patches),
        "n_old": len(old_discriminative_patches),
        "n": len(discriminative_patches),
        "n_same": n_same
    }
    record_dir = "./record"
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    np.save(os.path.join(record_dir, f"res_record_{iteration}.npy"),
            record_dic)
    if (iteration+1) % 1 == 0:
        torch.save({
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, os.path.join(record_dir, f"res_iteration.pth"))

    # if converge, stop or record
    
    if (n_same / max(len(old_discriminative_patches),
                     len(discriminative_patches)) > 0.95):
        torch.save({
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, os.path.join(record_dir, f"res_converged_{iteration}.pth"))
        break
    iteration += 1