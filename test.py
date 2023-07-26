import numpy as np
import os 
import utils


l = eval(utils.load_txt('./demo/demo.txt'))
print(l)
boxes = l[0]
print(boxes)
for box in boxes:
    print(box)
