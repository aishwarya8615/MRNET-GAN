import numpy as np
from matplotlib import pyplot as plt

axial = np.load("./dataset/MRNet-v1.0/train/axial/0000.npy")
coronal = np.load("./dataset/MRNet-v1.0/train/coronal/0000.npy")
sagittal = np.load("./dataset/MRNet-v1.0/train/sagittal/0000.npy")

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize = (15, 5))

ax1.imshow(axial[0, :, :], 'gray')
ax1.set_title('Case 0 | Slice 1 | axial')

ax2.imshow(coronal[0, :, :], 'gray')
ax2.set_title('Case 0 | Slice 1 | coronal')

ax3.imshow(sagittal[0, :, :], 'gray')
ax3.set_title('Case 0 | Slice 1 | sagittal')


ax4.imshow(axial[1, :, :], 'gray')
ax4.set_title('Case 0 | Slice 2 | axial')

ax5.imshow(coronal[1, :, :], 'gray')
ax5.set_title('Case 0 | Slice 2 | coronal')

ax6.imshow(sagittal[1, :, :], 'gray')
ax6.set_title('Case 0 | Slice 2 | sagittal')

plt.show()

