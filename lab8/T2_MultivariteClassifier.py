import math
import numpy as np

def Bayesian_classification(data, classes, sample):

    # prior probabilities
    values = list(data.values())
    total = len(values)
    p = []

    for v in classes:
        count = sum([1 for t in range(total) if values[t] == v])
        p.append(count / total)

    posterior = []
    L = []

    for i in range(len(classes)):

        # all samples belonging to class classes[i]
        temp_data = [key for key, value in data.items() if value == classes[i]]

        X = np.array(temp_data)            
        mean_vec = np.mean(X, axis=0)        
        cov_mat = np.cov(X.T)                # covariance matrix
        cov_inv = np.linalg.inv(cov_mat)
        det_cov = np.linalg.det(cov_mat)

        diff = np.array(sample) - mean_vec
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        const = 1 / (np.sqrt((2 * math.pi)**len(sample) * det_cov))

        likelihood = const * math.exp(exponent)
        L.append(likelihood)

        posterior.append(likelihood * p[i])

  
    evidence = sum(posterior)
    posterior_norm = [post / evidence for post in posterior]

    return classes[np.argmax(posterior_norm)]
