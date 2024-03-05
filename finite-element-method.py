# Ida Ciepiela
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

n = 100
h = 2 / n


def E(x):
    if x <= 1:
        return 2
    else:
        return 6


def e(i, x):
    if x <= h * (i - 1) or x >= h * (i + 1):
        return 0
    elif x < h * i:
        return x / h - i + 1
    else:
        return - x / h + i + 1


def e_der(i, x):
    if (x <= h * (i - 1)) or (x >= h * (i + 1)) or x == h * i:
        return 0
    elif x < h * i:
        return 1 / h
    else:
        return -1 / h


def u(x):
    return 3 * e(n, x)


def u_der(x):
    return 3 * e_der(n, x)


def b(i, j):
    start = max(0, h * (i - 1), h * (j - 1))
    stop = min(2, h * (i + 1), h * (j + 1))

    return -2 * e(i, 0) * e(j, 0) + \
        quad(lambda x: E(x) * e_der(i, x) * e_der(j, x), start, stop)[0]


# alternative version
# def b(i, j):
#     start = max(0, h * (i - 1), h * (j - 1))
#     stop = min(2, h * (i + 1), h * (j + 1))
#
#     return -4 * e(i, 0) * e(j, 0) + \
#         quad(lambda x: E(x) * e_der(i, x) * e_der(j, x), start, stop)[0]


def l(i):
    start = max(0, (i - 1) * h)
    stop = min(2, (i + 1) * h)
    return -20 * e(i, 0) - \
        quad(lambda x: E(x) * e_der(i, x) * u_der(x), start, stop)[0] + \
        2 * u(0) * e(i, 0)


# alternative version
# def l(i):
#
#     start = max(0, (i - 1) * h)
#     stop = min(2, (i + 1) * h)
#     return -20 * e(i, 0) - \
#         quad(lambda x: E(x) * e_der(i, x) * u_der(x), start, stop)[0] + \
#         4 * u(0) * e(i, 0) -\
#         quad(lambda x:1000*np.sin(x*np.pi) * e(i,x),start,stop)[0]


def main():
    left = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if abs(i - j) <= 1:
                left[i][j] = b(i, j)

    right = np.zeros(n)

    for i in range(n):
        right[i] = l(i)

    print(left)
    print(right)
    solution = np.linalg.solve(left, right)
    print(solution)

    fx = np.arange(0.0, 2.002, 0.002)
    fy = np.zeros(1001)
    for i in range(1001):
        for j in range(n):
            fy[i] += solution[j] * e(j, fx[i])
        fy[i] += u(fx[i])

    plt.plot(fx, fy)
    plt.show()


if __name__ == '__main__':
    main()
