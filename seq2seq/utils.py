import time


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))
