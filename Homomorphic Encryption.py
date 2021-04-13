import phe
import numpy as np

public_key, private_key = phe.generate_paillier_keypair(n_length=1024)

x = public_key.encrypt(5)

y = public_key.encrypt(3)

z = x + y

z_ = private_key.decrypt(z)

print(z_)

A = np.array([[1, 2, 3, 4, 5]])
print(A[:, 0])
