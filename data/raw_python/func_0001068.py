def cassist(self,dc,dt,dt2,nodiag=False,memlimit=-1):
	"""Calculates probability of gene i regulating gene j with continuous data assisted method,
	with the recommended combination of multiple tests.
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
	p:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)).
		Probability function from for recommended combination of multiple tests.
	For more information on tests, see paper.
	ftype can be found in auto.py.
	
	Example: see findr.examples.geuvadis5
	"""
	return _cassist_any(self,dc,dt,dt2,"pij_cassist",nodiag=nodiag,memlimit=memlimit)