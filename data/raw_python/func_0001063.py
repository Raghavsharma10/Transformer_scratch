def _gassist_any(self,dg,dt,dt2,name,na=None,nodiag=False,memlimit=-1):
	"""Calculates probability of gene i regulating gene j with genotype data assisted method,
	with the recommended combination of multiple tests.
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
	name:	actual C function name to call 
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
	"""
	if self.lib is None:
		raise ValueError("Not initialized.")
	import numpy as np
	from .auto import ftype_np,gtype_np
	from .types import isint
	if dg.dtype.char!=gtype_np:
		raise ValueError('Wrong input dtype for genotype data: dg.dtype.char is '+dg.dtype.char+'!='+gtype_np)
	if dt.dtype.char!=ftype_np or dt2.dtype.char!=ftype_np:
		raise ValueError('Wrong input dtype for gene expression data')
	if len(dg.shape)!=2 or len(dt.shape)!=2 or len(dt2.shape)!=2:
		raise ValueError('Wrong input shape')
	if type(nodiag) is not bool:
		raise ValueError('Wrong nodiag type')
	if not isint(memlimit):
		raise ValueError('Wrong memlimit type')
	if not (na is None or isint(na)):
		raise ValueError('Wrong na type')
	if na is not None and na<=0:
		raise ValueError('Input requires na>0.')
	ng=dg.shape[0]
	nt=dt2.shape[0]
	ns=dg.shape[1]
	nvx=na+1 if na else dg.max()+1
	nd=1 if nodiag else 0
	
	if nvx<2:
		raise ValueError('Invalid genotype values')
	if dt.shape!=dg.shape or dt2.shape[1]!=ns:
		raise ValueError('Wrong input shape')
	if np.isnan(dt).sum()+np.isnan(dt2).sum()>0:
		raise ValueError('NaN found.')

	func=self.cfunc(name,rettype='int',argtypes=['const MATRIXG*','const MATRIXF*','const MATRIXF*','MATRIXF*','size_t','byte','size_t'])
	d=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	dgr=np.require(dg,requirements=['A','C','O','W'])
	dtr=np.require(dt,requirements=['A','C','O','W'])
	dt2r=np.require(dt2,requirements=['A','C','O','W'])
	ret=func(dgr,dtr,dt2r,d,nvx,nd,memlimit)
	ans={'ret':ret,'p':d}
	return ans