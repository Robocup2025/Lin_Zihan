from pathlib import Path
import os
import random

li=random.sample(range(1,101),50)
for i in li:
    old=Path("img")/f"{i}.txt"
    new=old.with_suffix(".jpg")
    old.rename(new)