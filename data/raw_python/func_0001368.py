def load_geuvadis_data():
	"""This function loads downsampled data files from the Geuvadis study (Lappalainen, T. et al. Transcriptome and genome sequencing uncovers functional variation in humans. Nature 501, 506-511 (2013)), including expression levels of 10 miRNAs and 3000 genes for 360 European individuals. Among them, all miRNAs and 1000 genes have significant cis-eQTLs, whose haplotypes are also included. File data formats follow Findr's binary interface input/output requirement. A description of each file is available below:
	dmi.dat:	Expression levels of 10 miRNAs
	dgmi.dat:	Haplotypes of cis-eQTLs of 10 miRNAs
	dc.dat:		Continuous causal anchors for demonstration purposes, simulated from adding continuous noise to dgmi.dat
	dt.dat:		Expression levels of 1000 genes that have cis-eQTLs
	dt2.dat:	Expression levels of 3000 genes
	dgt.dat:	Haplotypes of cis-eQTLs of 1000 genes
	namest.txt:	3000 gene names"""
	from os.path import dirname,join
	from .auto import gtype_np,ftype_np
	import numpy as np
	def getdata(name,dtype,shape):
		d=join(dirname(__file__),'data','geuvadis',name)
		d=np.fromfile(d,dtype=dtype)
		d=d.reshape(*shape)
		return d
	
	ans={'dc':getdata('dc.dat',ftype_np,(10,360)),
	'dgmi':getdata('dgmi.dat',gtype_np,(10,360)),
	'dmi':getdata('dmi.dat',ftype_np,(10,360)),
	'dgt':getdata('dgt.dat',gtype_np,(1000,360)),
	'dt':getdata('dt.dat',ftype_np,(1000,360)),
	'dt2':getdata('dt2.dat',ftype_np,(3000,360))}
	
	f=open(join(dirname(__file__),'data','geuvadis','namest.txt'),'r')
	namest=[x.strip('\r\n') for x in f.readlines()]
	f.close()
	ans['namest']=namest
	return ans