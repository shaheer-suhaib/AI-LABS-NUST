import math
def Bayesian_classification( data,   classes,  sample,type):

    # prior probabilites
    count = 0
    p=[]
    for i in range(classes):
       count = sum([1  for j in len(data)  if data[j]== classes[i] ])
        
       p.append(count/len(data))

    # class parameters mean varience  covarience
    u= []
    L = []
    var = []
    posterior = []
   
    for i in len(classes):
        # all samples belonging to v in data
        # mean
        temp_data = [data[j]  for j in range(len(data)) if data[j]==classes[i]]
        u[i] = sum(temp_data)/len(temp_data)

        if type == "univariate":
           var.append( sum([(value - u[i])**2 for value in temp_data]) / len(temp_data))
        if type == "multivariate":

            pass
        # likelihood
        L[i] = (1/math.sqrt(2*math.pi*var[i])*math.exp(-((sample - u[i])**2)/(2*var[i])) )
        # posteriri
        posterior.append(L[i]*p[i])

    

    evidence = sum(posterior)

    posterior = [v/evidence for v in posterior]

    #prediction
    if posterior[]













        
        
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
    22: 1,
    17: None
}
