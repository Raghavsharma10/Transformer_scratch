def bootstrap(v):
	"""
Constructs Monte Carlo simulated data set using the
Bootstrap algorithm.                                                                                   

Usage:

>>> bootstrap(x)

where x is either an array or a list of arrays. If it is a
list, the code returns the corresponding list of bootstrapped 
arrays assuming that the same position in these arrays map the 
same "physical" object.

Rodrigo Nemmen, http://goo.gl/8S1Oo
	"""
	if type(v)==list:
		vboot=[]	# list of boostrapped arrays
		n=v[0].size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
		for x in v:	vboot.append(x[iran])
	else:	# if v is an array, not a list of arrays
		n=v.size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
		vboot=v[iran]
	
	return vboot