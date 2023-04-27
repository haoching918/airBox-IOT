import os

if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./data/device_week"):
    os.makedirs("./data/device_week")
if not os.path.exists("./data/discretized_device_week"):
    os.makedirs("./data/discretized_device_week")
if not os.path.exists("./data/time_week"):
    os.makedirs("./data/time_week")
    