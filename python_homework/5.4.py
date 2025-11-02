from pathlib import Path
import os

Path("img").mkdir(exist_ok=True)
for i in range(1, 101):
    (Path("img")/f"{i}.txt").touch()
    
