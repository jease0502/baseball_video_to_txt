from workflow import Workflow
import os

workflow = Workflow()

path = "/data/Projects/baseball/20211001/img"

count_file = 0
for i in os.listdir(path):
    count_file += 1

for i in range(count_file):
    workflow.run(os.path.join(path,str(i) , str(1) + ".png"))
    workflow.run(os.path.join(path, str(i)  , str(2) + ".png"))