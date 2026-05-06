def rank_pv(self,dt,dt2,memlimit=-1):
	"""Calculates p-values of gene i correlating with gene j by converting log likelihoods into probabilities per A for all B.
	dt:	numpy.ndarray(nt,ns,dtype=ftype(='=f4' by default)) Gene expression data for A
		Entry dt[i,j] is gene i's expression level for sample j.
	dt2:numpy.ndarray(nt2,ns,dtype=ftype(='=f4' by default)) Gene expression data for B.
		dt2 has the same format as dt, and can be identical with, different from, a subset of, or a superset of dt.
	memlimit:	The approximate memory usage limit in bytes for the library.  For datasets require a larger memory, calculation will fail with an error message. memlimit=0 defaults to unlimited memory usage.
	Return:	dictionary with following keys:
	ret:0 iff execution succeeded.
	p:	numpy.ndarray((nt,nt2),dtype=ftype(='=f4' by default)). P-values for A--B.
	ftype and gtype can be found in auto.py.
	
	Example: see findr.examples.geuvadis1 (similar format)
	"""
	if self.lib is None:
		raise ValueError("Not initialized.")
	import numpy as np
	from .auto import ftype_np,gtype_np
	from .types import isint
	if dt.dtype.char!=ftype_np or dt2.dtype.char!=ftype_np:
		raise ValueError('Wrong input dtype for gene expression data')
	if len(dt.shape)!=2 or len(dt2.shape)!=2:
		raise ValueError('Wrong input shape')
	if not isint(memlimit):
		raise ValueError('Wrong memlimit type')
	ng=dt.shape[0]
	nt=dt2.shape[0]
	ns=dt.shape[1]
	
	if dt2.shape[1]!=ns:
		raise ValueError('Wrong input shape')
	if np.isnan(dt).sum()+np.isnan(dt2).sum()>0:
		raise ValueError('NaN found.')

	dp=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	dtr=np.require(dt,requirements=['A','C','O','W'])
	dt2r=np.require(dt2,requirements=['A','C','O','W'])
	arglist=['const MATRIXF*','const MATRIXF*','MATRIXF*','size_t']
	args=[dtr,dt2r,dp,memlimit]
	func=self.cfunc('pij_rank_pv',rettype='int',argtypes=arglist)
	ret=func(*args)
	ans={'ret':ret,'p':dp}
	return ans