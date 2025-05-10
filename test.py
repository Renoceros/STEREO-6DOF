import time
now = []
now.append(time.time())
for i in range(20):
    time.sleep(2)
    now.append(time.time())
    print(f"time it took {i+1}: {int(now[i+1]-now[i])}s")