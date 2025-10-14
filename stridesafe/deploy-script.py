import numpy as np
import matplotlib.pyplot as plt

results_arr = np.array(results)  
size = int(np.sqrt(len(results_arr)))     

plt.figure(figsize=(3,3))
plt.imshow(1 - results_arr.reshape((size, size)), cmap="gray")
plt.axis('off')
plt.show()