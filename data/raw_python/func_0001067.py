def _cassist_any(self,dc,dt,dt2,name,nodiag=False,memlimit=-1):
	"""Calculates probability of gene i regulating gene j with continuous data assisted method,
	with the recommended combination of multiple tests.
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
	name:	actual C function name to call 
	nodiag:	skip diagonal regulations, i.e. regulation A->B for A=B.
		This should be set to True when A is a subset of B and aligned correspondingly.
	memlimit:	The approximate memory usage limit in bytes for the library.  For datasets require a larger memory, calculation will be split into smaller chunks. If the memory limit is smaller than minimum required, calculation can fail with an error message. memlimit=0 defaults to unlimited memory usage.
	Return:	dictionary with following keys:
	ret:0 iff execution succeeded.
	p:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)).
		Probability function from for recommended combination of multiple tests.
	For more information on tests, see paper.
	ftype can be found in auto.py.
	"""
	if self.lib is None:
		raise ValueError("Not initialized.")
	import numpy as np
	from .auto import ftype_np
	from .types import isint
	if dc.dtype.char!=ftype_np or dt.dtype.char!=ftype_np or dt2.dtype.char!=ftype_np:
		raise ValueError('Wrong input dtype for gene expression data')
	if len(dc.shape)!=2 or len(dt.shape)!=2 or len(dt2.shape)!=2:
		raise ValueError('Wrong input shape')
	if type(nodiag) is not bool:
		raise ValueError('Wrong nodiag type')
	if not isint(memlimit):
		raise ValueError('Wrong memlimit type')
	ng=dc.shape[0]
	nt=dt2.shape[0]
	ns=dc.shape[1]
	nd=1 if nodiag else 0
	
	if dt.shape!=dc.shape or dt2.shape[1]!=ns:
		raise ValueError('Wrong input shape')
	if np.isnan(dc).sum()+np.isnan(dt).sum()+np.isnan(dt2).sum()>0:
		raise ValueError('NaN found.')

	func=self.cfunc(name,rettype='int',argtypes=['const MATRIXF*','const MATRIXF*','const MATRIXF*','MATRIXF*','byte','size_t'])
	d=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	dcr=np.require(dc,requirements=['A','C','O','W'])
	dtr=np.require(dt,requirements=['A','C','O','W'])
	dt2r=np.require(dt2,requirements=['A','C','O','W'])
	ret=func(dcr,dtr,dt2r,d,nd,memlimit)
	ans={'ret':ret,'p':d}
	return ans