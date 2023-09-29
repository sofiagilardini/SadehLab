import numpy as np
import matplotlib.pyplot as plt

neuraldata_session3_tr_st_sh = np.load('./information/neuraldata_st_shuffled/A118_s3_neuraldata_tr_st_sh.npy', allow_pickle=True)

plt.imshow(neuraldata_session3_tr_st_sh)
plt.show()