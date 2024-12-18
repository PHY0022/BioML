import time
import os


def format_time(time_sec):
    h = time_sec // 3600
    m = (time_sec % 3600) // 60
    s = time_sec % 60
    return f"{int(h)}h {int(m)}m {int(s)}s"



def get_model_paths(result_dir):
    items = os.listdir(result_dir)
    # print(items)

    models = {}
    for item in items:
        if item.endswith(".model") or item.endswith(".h5"):
            # print(item)
            models[item.rsplit(".", 1)[0]] = os.path.join(result_dir, item)
    
    # print(models)
    return models



def get_params(params_path):
    with open(os.path.join(params_path, "params.txt"), "r") as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        if line.index("=") != -1:
            key, value = line.split("=")
            params[key] = value.strip()

    return params



if __name__ == "__main__":
    time_start = time.time()
    time.sleep(2)
    print(format_time(time.time() - time_start))
    # 0h 0m 2s

    print(format_time(12345))
    # 3h 25m 45s

    result_dir = r'D:\Programming\Python\113_BioML\exp\WE_with_DL\result-20241201024906'.replace('\\', "/")
    get_model_paths(result_dir)

    print(get_params(result_dir))