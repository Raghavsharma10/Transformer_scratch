def gassist(self,dg,dt,dt2,na=None,nodiag=False,memlimit=-1):
	"""Calculates probability of gene i regulating gene j with genotype data assisted method,
	with the recommended combination of multiple tests.
	Probabilities are converted from likelihood ratios separately for each A. This gives better
	predictions when the number of secondary targets (dt2) is large. (Check program warnings.)
	dg:	numpy.ndarray(nt,ns,dtype=gtype(='u1' by default)) Genotype data.
		Entry dg[i,j] is genotype i's value for sample j.
		Each value must be among 0,1,...,na.
		Genotype i must be best (and significant) eQTL of gene i (in dt).
	dt:	numpy.ndarray(nt,ns,dtype=ftype(='=f4' by default)) Gene expression data for A
		Entry dt[i,j] is gene i's expression level for sample j.
		Genotype i (in dg) must be best (and significant) eQTL of gene i.
	dt2:numpy.ndarray(nt2,ns,dtype=ftype(='=f4' by default)) Gene expression data for B.
		dt2 has the same format as dt, and can be identical with, different from, or a superset of dt.
		When dt2 is a superset of (or identical with) dt, dt2 must be arranged
		to be identical with dt at its upper submatrix, i.e. dt2[:nt,:]=dt, and
		set parameter nodiag = 1.
	na:	Number of alleles the species have. It determintes the maximum number of values each genotype can take. When unspecified, it is automatically
		determined as the maximum of dg.
	nodiag:	skip diagonal regulations, i.e. regulation A->B for A=B.
		This should be set to True when A is a subset of B and aligned correspondingly.
	memlimit:	The approximate memory usage limit in bytes for the library.  For datasets require a larger memory, calculation will be split into smaller chunks. If the memory limit is smaller than minimum required, calculation can fail with an error message. memlimit=0 defaults to unlimited memory usage.
	Return:	dictionary with following keys:
	ret:0 iff execution succeeded.
	p:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)).
		Probability function from for recommended combination of multiple tests.
	For more information on tests, see paper.
	ftype and gtype can be found in auto.py.
	
	Example: see findr.examples.geuvadis2, findr.examples.geuvadis3
	"""
	return _gassist_any(self,dg,dt,dt2,"pij_gassist",na=na,nodiag=nodiag,memlimit=memlimit)