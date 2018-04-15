#!/usr/bin/env python

import subprocess
import atexit
import time

ps = []

for i in range (1):
    ps.append(subprocess.Popen(['python', 'agent.py', '--visualize']))

for i in range (24):
    ps.append(subprocess.Popen(['python', 'agent.py']))

# for i in range (12):
#     ps.append(subprocess.Popen(['python', 'agent.py', '--explore']))

# for i in range (3):
#     ps.append(subprocess.Popen(['python', 'agent.py']))

def on_exit():
    for p in ps:
        p.kill ()

atexit.register(on_exit)

while True:
    time.sleep(60)
