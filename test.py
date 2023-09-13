import numpy as np
import os 
import utils_


l = eval(utils_.load_txt('./demo/demo.txt'))
print(l)
boxes = l[0]
print(boxes)
for box in boxes:
    print(box)
