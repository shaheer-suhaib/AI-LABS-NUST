import math

def Bayesian_classification(data, classes, sample, type):

    # prior probabilities
    p = []
    values = list(data.values())         
    total = len(values)

    print("\n=== STEP 1: PRIOR PROBABILITIES ===")

    for v in classes:
        count = sum([1 for j in range(total) if values[j] == v])
        p_val = count / total
        p.append(p_val)
        print(f"P(class={v}) = {p_val}")

    # class parameters mean variance 
    u = []
    var = []
    posterior = []
    L = []

    print("\n=== STEP 2: CLASS PARAMETERS & LIKELIHOOD ===")

    for i in range(len(classes)):

        # all samples
        temp_data = [key for key, value in data.items() if value == classes[i]]

        print(f"\nClass {classes[i]} data = {temp_data}")

        # mean
        mean = sum(temp_data) / len(temp_data)
        u.append(mean)
        print(f"Mean (u[{i}]) = {mean}")

        if type == "univariate":
            variance = sum([(value - u[i]) ** 2 for value in temp_data]) / len(temp_data)
            var.append(variance)
            print(f"Variance (var[{i}]) = {variance}")

            # likelihood
            likelihood = (1 / math.sqrt(2 * math.pi * var[i])) * \
                         math.exp(-((sample - u[i]) ** 2) / (2 * var[i]))

            L.append(likelihood)
            print(f"Likelihood P(sample={sample} | class={classes[i]}) = {likelihood}")

        if type == "multivariate":
            
            pass

        # unnormalized posterior
        post_val = L[i] * p[i]
        posterior.append(post_val)
        print(f"Unnormalized Posterior[{i}] = Likelihood * Prior = {post_val}")

    # evidence
    evidence = sum(posterior)
    print("\n=== STEP 3: EVIDENCE ===")
    print(f"P(sample) = {evidence}")

    # normalize
    posterior_norm = [v / evidence for v in posterior]
    print("\n=== STEP 4: NORMALIZED POSTERIOR ===")
    for i in range(len(classes)):
        print(f"P(class={classes[i]} | sample={sample}) = {posterior_norm[i]}")

    # prediction
    print("\n=== FINAL DECISION ===")
    if posterior_norm[0] >= posterior_norm[1]:
        print("This sample belongs to class 0")
    else:
        print("This sample belongs to class 1")


data = {
    10: 0,
    12: 0,
    9: 0,
    11: 0,
    13.5: 0,
    20: 1,
    18: 1,
    21: 1,
    19.5: 1,
    22: 1
}

Bayesian_classification(data, [0, 1], 17, "univariate")
