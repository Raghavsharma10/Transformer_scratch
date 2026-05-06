def bces(y1,y1err,y2,y2err,cerr):
	"""
Does the entire regression calculation for 4 slopes:
OLS(Y|X), OLS(X|Y), bisector, orthogonal.
Fitting form: Y=AX+B.

Usage:

>>> a,b,aerr,berr,covab=bces(x,xerr,y,yerr,cov)

Output:

- a,b : best-fit parameters a,b of the linear regression 
- aerr,berr : the standard deviations in a,b
- covab : the covariance between a and b (e.g. for plotting confidence bands)

Arguments:

- x,y : data
- xerr,yerr: measurement errors affecting x and y
- cov : covariance between the measurement errors
(all are arrays)

v1 Mar 2012: ported from bces_regress.f. Added covariance output.
Rodrigo Nemmen, http://goo.gl/8S1Oo
	"""
	# Arrays holding the code main results for each method:
	# Elements: 0-Y|X, 1-X|Y, 2-bisector, 3-orthogonal
	a,b,avar,bvar,covarxiz,covar_ba=numpy.zeros(4),numpy.zeros(4),numpy.zeros(4),numpy.zeros(4),numpy.zeros(4),numpy.zeros(4)
	# Lists holding the xi and zeta arrays for each method above
	xi,zeta=[],[]
	
	# Calculate sigma's for datapoints using length of conf. intervals
	sig11var = numpy.mean( y1err**2 )
	sig22var = numpy.mean( y2err**2 )
	sig12var = numpy.mean( cerr )
	
	# Covariance of Y1 (X) and Y2 (Y)
	covar_y1y2 = numpy.mean( (y1-y1.mean())*(y2-y2.mean()) )

	# Compute the regression slopes
	a[0] = (covar_y1y2 - sig12var)/(y1.var() - sig11var)	# Y|X
	a[1] = (y2.var() - sig22var)/(covar_y1y2 - sig12var)	# X|Y
	a[2] = ( a[0]*a[1] - 1.0 + numpy.sqrt((1.0 + a[0]**2)*(1.0 + a[1]**2)) ) / (a[0]+a[1])	# bisector
	if covar_y1y2<0:
		sign = -1.
	else:
		sign = 1.
	a[3] = 0.5*((a[1]-(1./a[0])) + sign*numpy.sqrt(4.+(a[1]-(1./a[0]))**2))	# orthogonal
	
	# Compute intercepts
	for i in range(4):
		b[i]=y2.mean()-a[i]*y1.mean()
	
	# Set up variables to calculate standard deviations of slope/intercept 
	xi.append(	( (y1-y1.mean()) * (y2-a[0]*y1-b[0]) + a[0]*y1err**2 ) / (y1.var()-sig11var)	)	# Y|X
	xi.append(	( (y2-y2.mean()) * (y2-a[1]*y1-b[1]) - y2err**2 ) / covar_y1y2	)	# X|Y
	xi.append(	xi[0] * (1.+a[1]**2)*a[2] / ((a[0]+a[1])*numpy.sqrt((1.+a[0]**2)*(1.+a[1]**2))) + xi[1] * (1.+a[0]**2)*a[2] / ((a[0]+a[1])*numpy.sqrt((1.+a[0]**2)*(1.+a[1]**2)))	)	# bisector
	xi.append(	xi[0] * a[3]/(a[0]**2*numpy.sqrt(4.+(a[1]-1./a[0])**2)) + xi[1]*a[3]/numpy.sqrt(4.+(a[1]-1./a[0])**2)	)	# orthogonal
	for i in range(4):
		zeta.append( y2 - a[i]*y1 - y1.mean()*xi[i]	)

	for i in range(4):
		# Calculate variance for all a and b
		avar[i]=xi[i].var()/xi[i].size
		bvar[i]=zeta[i].var()/zeta[i].size
		
		# Sample covariance obtained from xi and zeta (paragraph after equation 15 in AB96)
		covarxiz[i]=numpy.mean( (xi[i]-xi[i].mean()) * (zeta[i]-zeta[i].mean()) )
	
	# Covariance between a and b (equation after eq. 15 in AB96)
	covar_ab=covarxiz/y1.size
	
	return a,b,numpy.sqrt(avar),numpy.sqrt(bvar),covar_ab