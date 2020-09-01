import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputFile.txt")
plt.xlabel("generation")
plt.ylabel("makespan")
plt.plot(df["x"], df["y"])
plt.show()
