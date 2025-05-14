# timer.py

import time

_last_time = time.perf_counter()


def pintime(text: str):
    global _last_time
    current_time = time.perf_counter()
    delta = current_time - _last_time
    _last_time = current_time
    print(f"{delta:.5f} : {text}")
    return delta
