def jitEigh(A,maxTries=10,warning=True):
    """
    Do a Eigenvalue Decomposition with Jitter,

    works as jitChol
    """
    warning = True
    jitter = 0
    i = 0

    while(True):
        if jitter == 0:
            jitter = abs(SP.trace(A))/A.shape[0]*1e-6
            S,U = linalg.eigh(A)

        else:
            if warning:
                # pdb.set_trace()
		# plt.figure()
		# plt.imshow(A, interpolation="nearest")
		# plt.colorbar()
		# plt.show()
                logging.error("Adding jitter of %f in jitEigh()." % jitter)
            S,U = linalg.eigh(A+jitter*SP.eye(A.shape[0]))

        if S.min()>1E-10:
            return S,U

        if i<maxTries:
            jitter = jitter*10
        i += 1
            
    raise linalg.LinAlgError("Matrix non positive definite, jitter of " +  str(jitter) + " added but failed after " + str(i) + " trials.")