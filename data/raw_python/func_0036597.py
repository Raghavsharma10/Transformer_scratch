def smallest_prime_factor(Q):
    """Find the smallest number factorable by the small primes 2, 3, 4, and 7 
    that is larger than the argument Q"""

    A = Q;
    while(A != 1):
        if(np.mod(A, 2) == 0):
            A = A / 2
        elif(np.mod(A, 3) == 0):
            A = A / 3
        elif(np.mod(A, 5) == 0):
            A = A / 5
        elif(np.mod(A, 7) == 0):
            A = A / 7;
        else:
            A = Q + 1;
            Q = A;

    return Q