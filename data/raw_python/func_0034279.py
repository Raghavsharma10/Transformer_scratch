def randindex(lo, hi, n = 1.):
	"""
	Yields integers in the range [lo, hi) where 0 <= lo < hi.  Each
	return value is a two-element tuple.  The first element is the
	random integer, the second is the natural logarithm of the
	probability with which that integer will be chosen.

	The CDF for the distribution from which the integers are drawn goes
	as [integer]^{n}, where n > 0.  Specifically, it's

		CDF(x) = (x^{n} - lo^{n}) / (hi^{n} - lo^{n})

	n = 1 yields a uniform distribution;  n > 1 favours larger
	integers, n < 1 favours smaller integers.
	"""
	if not 0 <= lo < hi:
		raise ValueError("require 0 <= lo < hi: lo = %d, hi = %d" % (lo, hi))
	if n <= 0.:
		raise ValueError("n <= 0: %g" % n)
	elif n == 1.:
		# special case for uniform distribution
		try:
			lnP = math.log(1. / (hi - lo))
		except ValueError:
			raise ValueError("[lo, hi) domain error")
		hi -= 1
		rnd = random.randint
		while 1:
			yield rnd(lo, hi), lnP

	# CDF evaluated at index boundaries
	lnP = numpy.arange(lo, hi + 1, dtype = "double")**n
	lnP -= lnP[0]
	lnP /= lnP[-1]
	# differences give probabilities
	lnP = tuple(numpy.log(lnP[1:] - lnP[:-1]))
	if numpy.isinf(lnP).any():
		raise ValueError("[lo, hi) domain error")

	beta = lo**n / (hi**n - lo**n)
	n = 1. / n
	alpha = hi / (1. + beta)**n
	flr = math.floor
	rnd = random.random
	while 1:
		index = int(flr(alpha * (rnd() + beta)**n))
		# the tuple look-up provides the second part of the
		# range safety check on index
		assert index >= lo
		yield index, lnP[index - lo]