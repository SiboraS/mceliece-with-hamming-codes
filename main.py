import numpy as np
from math import ceil, log2


def generate_random_binary_matrix(size):
    while True:
        matrix = np.random.randint(2, size=(size, size), dtype=int)
        if np.linalg.matrix_rank(matrix) == size and np.linalg.det(matrix) != 0:
            return matrix


def generate_random_binary_permutation_matrix(size):
    while True:
        matrix = np.zeros((size, size), dtype=int)
        cols = np.arange(size)
        np.random.shuffle(cols)
        for i in range(size):
            matrix[i][cols[i]] = 1
        if np.array_equal(matrix @ matrix.T, np.eye(size)):  # Check if matrix is a permutation matrix
            return matrix


# Get values of n and k from the user
n = int(input("Enter the value of n: "))
k = int(input("Enter the value of k: "))

# Compute the number of parity bits required
r = ceil(log2(n+1))
print(f'r: {r}')

# Compute the positions of the parity bits
positions = [2**i-1 for i in range(r)]

# Construct the parity check matrix H
H = np.zeros((r, n), dtype=int)
for i, pos in enumerate(positions):
    H[i, pos] = 1
    for j in range(1, n+1):
        if (j >> i) & 1:
            H[i, (pos-j) % n] = 1


print("H:\n", H)

# Transform the check matrix H to standard form (AT | In-k)
for j in range(r):
    # Find the pivot row (the row with the first non-zero entry in the current column)
    pivot = np.argmax(H[j:, j]) + j

    # Swap the pivot row with the current row (if they are different)
    if pivot != j:
        H[[j, pivot], :] = H[[pivot, j], :]

    # Eliminate the entries below the pivot by performing row operations
    for i in range(r-1, j, -1):
        if H[j, i] == 1:
            H[i, :] = (H[i, :] + H[j, :]) % 2

    # Eliminate the entries above the pivot by performing row operations
    if j == r - 1:
        for i in range(j):
            if H[i, j] == 1:
                H[i, :] = (H[i, :] + H[j, :]) % 2

print(f'H standard:\n{H}')

# Find the AT matrix
AT = H[:, :k]

# Find the transpose of the AT matrix to get A
A = AT.T
print(f'A:\n {A}')

# Get the identity matrix for G (Ik)
I = np.eye(k, dtype=int)
print(f'I:\n {I}')

# Get the generator matrix G (Ik | A)
G = np.concatenate((I, A), axis=1)
print(f'G:\n {G}')

# Generate a random matrix S (non-degenerate)
S = generate_random_binary_matrix(k)
print(f'S:\n {S}')
# Generate a random permutation matrix P
P = generate_random_binary_permutation_matrix(n)
print(f'P:\n {P}')

# Public key
G1 = np.mod(S @ G @ P, 2)
print(f'G1:\n {G1}')

# Bob
message = input('Please enter your message: ')
m = np.array(list(map(int, message.split())))

# Generate a random error array with weight 1
e = np.zeros((1, n), dtype=int)
# Generate a random column index
j = np.random.randint(n)
# Set the element at the random column index to 1
e[0, j] = 1
print(f'e: {e}')

c = np.mod(np.dot(m, G1) + e, 2)
print(f'c: {c}')

# Alice
S_inv = np.mod(np.linalg.inv(S), 2)
print(f'S_inv:\n{S_inv}')

P_inv = P.T
print(f'P_inv:\n{P_inv}')

# c1 = np.mod(c @ P_inv - e @ P_inv, 2)
c1 = np.mod(c @ P_inv, 2)
print(f'c1: {c1}')

syndrome = np.mod(np.dot(H, c1.T), 2)
syndrome = syndrome.T
print(f'syndrome: {syndrome}')

# Find the error bit
result = np.all(H.T == syndrome, axis=1)
print(f'result: {result}')
index = np.where(result == True)[0]
print(f'index: {index}')

# Correct the error on the bit (index) given by the syndrome
wrong_bit = c1[0][index]
if wrong_bit == 1:
    c1[0][index] = 0
else:
    c1[0][index] = 1
print(f'c1 corrected: {c1}')

# Get the information bits from c1
mS = c1[0, :k]
print(f'mS: {mS}')

print(f'm (sent): {m}')
m_final = np.mod(mS @ S_inv, 2)
print(f'm (decoded): {m_final}')
