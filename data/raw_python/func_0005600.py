def bcesboot(y1,y1err,y2,y2err,cerr,nsim=10000):
	
	"""
	Does the BCES with bootstrapping.	
	
	Usage:
	
	>>> a,b,aerr,berr,covab=bcesboot(x,xerr,y,yerr,cov,nsim)
	
	:param x,y: data
	:param xerr,yerr: measurement errors affecting x and y
	:param cov: covariance between the measurement errors (all are arrays)
	:param nsim: number of Monte Carlo simulations (bootstraps)
	
	:returns: a,b -- best-fit parameters a,b of the linear regression 
	:returns: aerr,berr -- the standard deviations in a,b
	:returns: covab -- the covariance between a and b (e.g. for plotting confidence bands)
	
	.. note:: this method is definitely not nearly as fast as bces_regress.f. Needs to be optimized. Maybe adapt the fortran routine using f2python?
	
	v1 Mar 2012: ported from bces_regress.f. Added covariance output.
	Rodrigo Nemmen, http://goo.gl/8S1Oo
	"""
	
	
	# Progress bar initialization
	
	"""
	My convention for storing the results of the bces code below as 
	matrixes for processing later are as follow:
	
	      simulation\method  y|x x|y bisector orthogonal
	          sim0           ...
	Am =      sim1           ...
	          sim2           ...
	          sim3           ...
	"""
	for i in range(nsim):
		[y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])
		
		asim,bsim,errasim,errbsim,covabsim=bces(y1sim,y1errsim,y2sim,y2errsim,cerrsim)	
		
		if i==0:
			# Initialize the matrixes
			am,bm=asim.copy(),bsim.copy()
		else: 
			am=numpy.vstack((am,asim))
			bm=numpy.vstack((bm,bsim))
				
		# Progress bar
	
	# Bootstrapping results
	a=numpy.array([ am[:,0].mean(),am[:,1].mean(),am[:,2].mean(),am[:,3].mean() ])
	b=numpy.array([ bm[:,0].mean(),bm[:,1].mean(),bm[:,2].mean(),bm[:,3].mean() ])

	# Error from unbiased sample variances
	erra,errb,covab=numpy.zeros(4),numpy.zeros(4),numpy.zeros(4)
	for i in range(4):
		erra[i]=numpy.sqrt( 1./(nsim-1) * ( numpy.sum(am[:,i]**2)-nsim*(am[:,i].mean())**2 ))
		errb[i]=numpy.sqrt( 1./(nsim-1) * ( numpy.sum(bm[:,i]**2)-nsim*(bm[:,i].mean())**2 ))
		covab[i]=1./(nsim-1) * ( numpy.sum(am[:,i]*bm[:,i])-nsim*am[:,i].mean()*bm[:,i].mean() )
	
	return a,b,erra,errb,covab