# CSC311 Homework 1 question 1a
import numpy as np
import matplotlib.pyplot as plt


def mean_variance_euclidean(d):
    sample = [np.random.rand(d) for _ in range(100)]

    euclid_dist = []
    l1_dist = []
    for i in range(100):
        for k in range(i+1, 100):
            # if not np.array_equal(sample[i], sample[k]):
            diff = sample[i] - sample[k]
            euclid_d = np.dot(diff.T, diff)
            euclid_dist.append(euclid_d)

            l1 = np.sum([np.abs(sample[i][j] - sample[k][j])
                         for j in range(d)])
            l1_dist.append(l1)
    # print(euclid_dist)
    return np.mean(euclid_dist), np.std(euclid_dist), \
           np.mean(l1_dist), np.std(l1_dist)


if __name__ == "__main__":
    dimensions = np.array([2 ** i for i in range(11)])
    euclid_mean = []
    euclid_std = []
    l1_mean = []
    l1_std = []
    for d in dimensions:
        e_mean, e_std, l1_m, l1_v = mean_variance_euclidean(d)
        euclid_mean.append(e_mean)
        euclid_std.append(e_std)
        l1_mean.append(l1_m)
        l1_std.append(l1_v)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Euclidean
    ax1.set_ylabel("Average")
    ax1.set_xlabel("d")
    ax1.plot(dimensions, euclid_mean)

    ax2.set_ylabel("Standard Deviation")
    ax2.set_xlabel("d")
    ax2.plot(dimensions, euclid_std)
    fig.suptitle("Euclidean Distance")

    plt.show()

    # L1 distance
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_ylabel("Average")
    ax1.set_xlabel("d")
    ax1.plot(dimensions, l1_mean)

    ax2.set_ylabel("Standard Deviation")
    ax2.set_xlabel("d")
    ax2.plot(dimensions, l1_std)
    fig.suptitle("L1 Distance")

    plt.show()
