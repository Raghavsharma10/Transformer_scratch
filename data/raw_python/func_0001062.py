def gassists_pv(self,dg,dt,dt2,na=None,memlimit=-1):
	"""Calculates p-values of gene i regulating gene j with genotype data assisted method with multiple tests.
	dg:	numpy.ndarray(nt,ns,dtype=gtype(='u1' by default)) Genotype data.
		Entry dg[i,j] is genotype i's value for sample j.
		Each value must be among 0,1,...,na.
		Genotype i must be best (and significant) eQTL of gene i (in dt).
	dt:	numpy.ndarray(nt,ns,dtype=ftype(='=f4' by default)) Gene expression data for A
		Entry dt[i,j] is gene i's expression level for sample j.
		Genotype i (in dg) must be best (and significant) eQTL of gene i.
	dt2:numpy.ndarray(nt2,ns,dtype=ftype(='=f4' by default)) Gene expression data for B.
		dt2 has the same format as dt, and can be identical with, different from, or a superset of dt.
	na:	Number of alleles the species have. It determintes the maximum number of values each genotype can take. When unspecified, it is automatically
		determined as the maximum of dg.
	memlimit:	The approximate memory usage limit in bytes for the library.  For datasets require a larger memory, calculation will be split into smaller chunks. If the memory limit is smaller than minimum required, calculation can fail with an error message. memlimit=0 defaults to unlimited memory usage.
	Return:	dictionary with following keys:
	ret:0 iff execution succeeded.
	p1:	numpy.ndarray(nt,dtype=ftype(='f4' by default)). P-values for LLR of test 1.
		Test 1 calculates E(A)->A v.s. E(A)  A.
	p2:	numpy.ndarray((nt,nt2),dtype=ftype(='f4' by default)). P-values for LLR of test 2.
		Test 2 calculates E(A)->A--B with E(A)->B v.s. E(A)->A<-B.
	p3:	numpy.ndarray((nt,nt2),dtype=ftype(='f4' by default)). P-values for LLR of test 3.
		Test 3 calculates E(A)->A--B with E(A)->B v.s. E(A)->A->B.
	p4:	numpy.ndarray((nt,nt2),dtype=ftype(='f4' by default)). P-values for LLR of test 4.
		Test 4 calculates E(A)->A--B with E(A)->B v.s. E(A)->A  B.
	p5:	numpy.ndarray((nt,nt2),dtype=ftype(='f4' by default)). P-values for LLR of test 5.
		Test 5 calculates E(A)->A--B with E(A)->B v.s. B<-E(A)->A.
	For more information on tests, see paper.
	ftype and gtype can be found in auto.py.
	
	Example: see findr.examples.geuvadis6
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
	
	if nvx<2:
		raise ValueError('Invalid genotype values')
	if dt.shape!=dg.shape or dt2.shape[1]!=ns:
		raise ValueError('Wrong input shape')
	if np.isnan(dt).sum()+np.isnan(dt2).sum()>0:
		raise ValueError('NaN found.')
	
	arglist=['const MATRIXG*','const MATRIXF*','const MATRIXF*','VECTORF*','MATRIXF*','MATRIXF*','MATRIXF*','MATRIXF*','size_t','size_t']
	dgr=np.require(dg,requirements=['A','C','O','W'])
	dtr=np.require(dt,requirements=['A','C','O','W'])
	dt2r=np.require(dt2,requirements=['A','C','O','W'])
	d1=np.require(np.zeros(ng,dtype=dt.dtype),requirements=['A','C','O','W'])
	d2=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	d3=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	d4=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	d5=np.require(np.zeros((ng,nt),dtype=dt.dtype),requirements=['A','C','O','W'])
	args=[dgr,dtr,dt2r,d1,d2,d3,d4,d5,nvx,memlimit]
	func=self.cfunc('pijs_gassist_pv',rettype='int',argtypes=arglist)
	ret=func(*args)
	ans={'ret':ret,'p1':d1,'p2':d2,'p3':d3,'p4':d4,'p5':d5}
	return ans