import time


def format_time(time_sec):
    h = time_sec // 3600
    m = (time_sec % 3600) // 60
    s = time_sec % 60
    return f"{int(h)}h {int(m)}m {int(s)}s"

# time_start = time.time()
# time.sleep(2)
# print(format_time(time.time() - time_start))
# # 0h 0m 2s

# print(format_time(12345))
# # 3h 25m 45s