import pandas as pd

df = pd.read_csv("./outputFile.txt")
last = 10000000
for i in df.y:
    if i > last:
        print("error")
        break
    last = i

