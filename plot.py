import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputFile.txt")
plt.xlabel("generation")
plt.ylabel("value")
plt.plot(df["x"], df["quality"])
# plt.plot(df["x"], df["dead"])
# plt.plot(df["x"], df["toolate"])
# plt.plot(df["x"], df["makespan"])
plt.show()
