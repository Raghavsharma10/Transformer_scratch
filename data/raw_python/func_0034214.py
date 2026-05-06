def get_bestnr(self, index=4.0, nhigh=3.0, null_snr_threshold=4.25,\
		           null_grad_thresh=20., null_grad_val = 1./5.):
		"""
		Get the BestNR statistic for each row in the table
		"""
		return [row.get_bestnr(index=index, nhigh=nhigh,
		                       null_snr_threshold=null_snr_threshold,
		                       null_grad_thresh=null_grad_thresh,
		                       null_grad_val=null_grad_val)
		        for row in self]