import os
import re

import numpy as np
import matplotlib.pyplot as plt

gas_class = []
features_lists = [] 

with open('batch1.dat', 'r') as file_data:
  for i, line in enumerate(file_data.readlines()):
    gas_class.append(int(line[0]))
    rest = line[1:]
    rest.strip()
    features_lists.insert(i, list(map(float, re.findall('\s\d*:([\d.-]+)', rest))))

for i in range(2):
  #print(gas_class[i])
  print(features_lists[i][0])

gas_class = np.array(gas_class, dtype=np.float32)
features_matrix = np.array(features_lists, dtype=np.float32)
log_features_matrix = np.multiply(np.sign(features_matrix), np.log(np.abs(features_matrix) + 1))


plt.figure()
plt.subplot(2,2,1)
plt.plot(np.multiply(np.sign(features_matrix[:,4]), np.log(np.abs(features_matrix[:, 4]) + 1)))
plt.subplot(2,2,2)
plt.plot(np.multiply(np.sign(features_matrix[:,5]), np.log(np.abs(features_matrix[:, 5]) + 1)))
plt.subplot(2,2,3)
plt.plot(np.multiply(np.sign(features_matrix[:,6]), np.log(np.abs(features_matrix[:, 6]) + 1)))
plt.subplot(2,2,4)
plt.plot(np.multiply(np.sign(features_matrix[:,7]), np.log(np.abs(features_matrix[:, 7]) + 1)))
plt.show()
