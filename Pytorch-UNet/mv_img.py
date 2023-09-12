import shutil 
import os 

start = 00000
end = 10000

if __name__ == "__main__":
    os.mkdir("data/imgs_batch/")
    os.mkdir("data/label_kp_batch/")
    for i in range(start,end):
        src_fn = f"data/imgs/img_{i}.png"
        dest_fn = f"data/imgs_batch/img_{i}.png"
        if os.path.exists(src_fn):
            shutil.copyfile(src_fn, dest_fn)
        else:
            break
        
        for j in range(14):
            src_fn = f"data/label_kp/img_{i}_{j}.png"
            dest_fn = f"data/label_kp_batch/img_{i}_{j}.png"
            if os.path.exists(src_fn):
                shutil.copyfile(src_fn, dest_fn)
