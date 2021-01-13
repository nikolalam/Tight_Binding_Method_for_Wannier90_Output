import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.process_time()
num_wan = 36
basis_vector = [[5.15, 0, 0], [0, 5.15, 0], [0, 0, 5.15]]
k_mesh = 21
E_fermi = 4.6579
K_point_path = [[0.5, 0.5, 0.5],
                [0, 0, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
                [0, 0, 0]  ]

Symmetry_point_label1 = "R"
Symmetry_point_label2 = "G"
Symmetry_point_label3 = "X"
Symmetry_point_label4 = "M"
Symmetry_point_label5 = "G"

lower_bound = -1.2
upper_bound = 1.2

V = np.dot(basis_vector[0], np.cross(basis_vector[1], basis_vector[2]) )
rec = [np.cross(basis_vector[1], basis_vector[2]) * 2 * np.pi / V,
       np.cross(basis_vector[2], basis_vector[0]) * 2 * np.pi / V,
       np.cross(basis_vector[0], basis_vector[1]) * 2 * np.pi / V]


for i in range(len(K_point_path)):
    K_point_path[i] = K_point_path[i][0] * rec[0] + K_point_path[i][1] * rec[1] + K_point_path[i][2] * rec[2]

with open("wannier90_hr.dat", "r") as f:
    lines = f.readlines()
    f.close()


def k_path():
    k_point = []
    for i in range(len(K_point_path) - 1):
        interval = np.array(K_point_path[i + 1]) - np.array(K_point_path[i])
        interval = interval / k_mesh
        for j in range(k_mesh + 1):
            k_point.append(np.array(K_point_path[i]) + j * interval)
    return k_point


def phase(R1, R2, R3, k1, k2, k3):
    R1_vector = R1 * np.array(basis_vector[0])
    R2_vector = R2 * np.array(basis_vector[1])
    R3_vector = R3 * np.array(basis_vector[2])
    R_vec = R1_vector + R2_vector + R3_vector
    inner_product = np.dot(R_vec, [k1, k2, k3])
    return np.exp(1j * inner_product)


def matrix_element():
    factor = []
    R = []

    for i in range(num_wan):
        factor.append([])
        R.append([])
        for j in range(num_wan):
            factor[len(factor) - 1].append([])
            R[len(R) - 1].append([])

    for i in range(len(lines)):
        if len(lines[i].split()) == 7:
            factor[int(lines[i].split()[3]) - 1][int(lines[i].split()[4]) - 1].append(
                float(lines[i].split()[5]) + 1j * float(lines[i].split()[6]))
            R[int(lines[i].split()[3]) - 1][int(lines[i].split()[4]) - 1].append(
                [float(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2])])
    return factor, R


def matrix_construct(factor, R, k1, k2, k3):
    H = np.zeros((num_wan, num_wan),dtype='complex')
    for i in range(num_wan):
        for j in range(num_wan):
            for k in range(len(R[i][j])):
                H[i][j] = H[i][j] + factor[i][j][k] * phase( R[i][j][k][0], R[i][j][k][1], R[i][j][k][2], k1, k2, k3)
    return H


def run():
    solution = []
    for i in range(num_wan):
        solution.append([])

    k_line = k_path()
    print('Constructing the matrix')
    factor, R = matrix_element()

    for l in range(len(k_line)):
        H = matrix_construct(factor, R, k_line[l][0], k_line[l][1], k_line[l][2])
        eig = np.linalg.eigvals(H)
        idx = np.argsort(eig)
        eig = eig[idx]
        for i in range(len(eig)):
            solution[i].append(eig[i] - E_fermi)
        print("Process Finished ", l * 100 / len(k_line), '%')

    return solution


solution = run()

ax = plt.axes()
for i in range(len(solution)):
    ax.plot(range(len(solution[i])), solution[i], color='black')
plt.ylim(lower_bound,upper_bound)
plt.plot([0,len(solution[0])],[0,0],color='black')
plt.grid(True)
plt.ylabel(r"$E - E_{fermi}$"' (eV)')
plt.xlabel("Wave vector  "r"$\vec{k}$")

ax.xaxis.set_major_locator(plt.MultipleLocator(k_mesh+1))

def format_func(N,ticks):
    if N == 0:
        return Symmetry_point_label1
    elif N == (k_mesh + 1) :
        return Symmetry_point_label2
    elif N == 2 * (k_mesh + 1) :
        return Symmetry_point_label3
    elif N == 3 * (k_mesh + 1) :
        return Symmetry_point_label4
    elif N == 4 *( k_mesh + 1) :
        return Symmetry_point_label5

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

end_time = time.process_time()

print("Process Finished")
print('CPU Excution time (**mins)  =', (end_time - start_time) / 60)
