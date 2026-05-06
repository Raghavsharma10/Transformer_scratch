def ab(x):
	"""
This method is the big bottleneck of the parallel BCES code. That's the 
reason why I put these calculations in a separate method, in order to 
distribute this among the cores. In the original BCES method, this is 
inside the main routine.
	
Argument:
[y1,y1err,y2,y2err,cerr,nsim]
where nsim is the number of bootstrapping trials sent to each core.

:returns: am,bm : the matrixes with slope and intercept where each line corresponds to a bootrap trial and each column maps a different BCES method (ort, y|x etc).

Be very careful and do not use lambda functions when calling this 
method and passing it to multiprocessing or ipython.parallel!
I spent >2 hours figuring out why the code was not working until I
realized the reason was the use of lambda functions.
	"""
	y1,y1err,y2,y2err,cerr,nsim=x[0],x[1],x[2],x[3],x[4],x[5]
	
	for i in range(nsim):
		[y1sim,y1errsim,y2sim,y2errsim,cerrsim]=bootstrap([y1,y1err,y2,y2err,cerr])

		asim,bsim,errasim,errbsim,covabsim=bces(y1sim,y1errsim,y2sim,y2errsim,cerrsim)	
	
		if i==0:
			# Initialize the matrixes
			am,bm=asim.copy(),bsim.copy()
		else: 
			am=numpy.vstack((am,asim))
			bm=numpy.vstack((bm,bsim))
		
	return am,bm