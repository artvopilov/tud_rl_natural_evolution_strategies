import numpy as np
from scipy import linalg as sl


a = np.random.rand(2, 2) * 100 - 50
print(a)
inva = np.linalg.pinv(a)
print(inva)
sqrtm_inva = sl.sqrtm(inva)
print(sqrtm_inva)

print(np.dot(sqrtm_inva.T, sqrtm_inva))
