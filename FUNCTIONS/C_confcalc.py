import scipy.stats as st
import math

def errortype(n, acc, confidence=0.95, CI=False): 
    se = math.sqrt((acc*(1-acc))/n)
    z= 1 - ((1 - confidence) / 2)
    if CI:
        error = z * se
    else:
        error = se
    return error



