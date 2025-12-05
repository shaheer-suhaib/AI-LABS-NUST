import math
import numpy as np

def naive_bayes(data, classes, sample):
    
    
    sample = np.array(sample)

    values = list(data.values())
    total = len(values)

    print("\n===  1: prior probabilites ===")
    priors = []
    for c in classes:
        count = values.count(c)
        p = count / total
        priors.append(p)
        print(f"P(class={c}) = {p}")

   
    posteriors = []

    print("\n===  2: CLASS parameters & likelihood ===")
    for i, c in enumerate(classes):

        class_points = np.array([key for key, val in data.items() if val == c])
        print(f"\nClass {c} data = {class_points}")

        mean_vec = np.mean(class_points, axis=0)
        var_vec  = np.var(class_points, axis=0)

        print(f"Mean = {mean_vec}")
        print(f"Variance = {var_vec}")

        
        likelihood_dims = (1 / np.sqrt(2*np.pi*var_vec)) * np.exp(-((sample - mean_vec)**2) / (2*var_vec))
        likelihood = np.prod(likelihood_dims)

        print(f"Likelihood = {likelihood}")

        post = likelihood * priors[i]
        posteriors.append(post)
        print(f"Unnormalized posterior = {post}")

    
    evidence = sum(posteriors)

    print("\n=== STEP 3: NORMALIZED POSTERIORS ===")
    for i, c in enumerate(classes):
        posteriors[i]=  posteriors[i] / evidence
        print(posteriors[i])
        print( )

    
    prediction = classes[np.argmax(posteriors)]
    print("\n=== FINAL DECISION ===")
    print(f"Predicted class = {prediction}")
    return prediction  

data = {
    (25, 40000): 0,
    (35, 60000): 0,
    (45, 80000): 0,
    (20, 20000): 0,
    (35, 120000): 0,
    (52, 18000): 0,
    (23, 95000): 1,
    (40, 62000): 1,
    (60, 100000): 1,
    (48, 220000): 1,
    (33, 150000): 1,
    
}

naive_bayes(data,[0,1],(48,142000))