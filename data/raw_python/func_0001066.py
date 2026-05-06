def cassists(self,dc,dt,dt2,nodiag=False,memlimit=-1):
	"""Calculates probability of gene i regulating gene j with continuous data assisted method,
	with multiple tests, by converting log likelihoods into probabilities per A for all B.
	Probabilities are converted from likelihood ratios separately for each A. This gives better
	predictions when the number of secondary targets (dt2) is large. (Check program warnings.)
	dc:	numpy.ndarray(nt,ns,dtype=ftype(='f4' by default)) Continuous anchor data.
		Entry dc[i,j] is anchor i's value for sample j.
		Anchor i is used to infer the probability of gene i -> any other gene.
	dt:	numpy.ndarray(nt,ns,dtype=ftype(='=f4' by default)) Gene expression data for A
		Entry dt[i,j] is gene i's expression level for sample j.
	dt2:numpy.ndarray(nt2,ns,dtype=ftype(='=f4' by default)) Gene expression data for B.
		dt2 has the same format as dt, and can be identical with, different from, or a superset of dt.
		When dt2 is a superset of (or identical with) dt, dt2 must be arranged
		to be identical with dt at its upper submatrix, i.e. dt2[:nt,:]=dt, and
		set parameter nodiag = 1.
	nodiag:	skip diagonal regulations, i.e. regulation A->B for A=B.
		This should be set to True when A is a subset of B and aligned correspondingly.
	memlimit:	The approximate memory usage limit in bytes for the library.  For datasets require a larger memory, calculation will be split into smaller chunks. If the memory limit is smaller than minimum required, calculation can fail with an error message. memlimit=0 defaults to unlimited memory usage.
	Return:	dictionary with following keys:
	ret:0 iff execution succeeded.
	p1:	numpy.ndarray(nt,dtype=ftype(='=f4' by default)). Probability for test 1.
		Test 1 calculates E(A)->A v.s. E(A)  A. The earlier one is preferred.
		For nodiag=False, because the function expects significant anchors, p1 always return 1.
		For nodiag=True, uses diagonal elements of p2.
	p2:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)). Probability for test 2.
		Test 2 calculates E(A)->A--B with E(A)->B v.s. E(A)->A<-B. The earlier one is preferred.
	p3:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)). Probability for test 3.
		Test 3 calculates E(A)->A--B with E(A)->B v.s. E(A)->A->B. The latter one is preferred.
	p4:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)). Probability for test 4.
		Test 4 calculates E(A)->A--B with E(A)->B v.s. E(A)->A  B. The earlier one is preferred.
	p5:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)). Probability for test 5.
		Test 5 calculates E(A)->A--B with E(A)->B v.s. B<-E(A)->A. The earlier one is preferred.
	For more information on tests, see paper.
	ftype can be found in auto.py.
	
	Example: see findr.examples.geuvadis4 (similar format)
	"""
	return _cassists_any(self,dc,dt,dt2,"pijs_cassist",nodiag=nodiag,memlimit=memlimit)