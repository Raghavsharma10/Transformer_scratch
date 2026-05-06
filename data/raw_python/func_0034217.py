def get_bestnr(self, index=4.0, nhigh=3.0, null_snr_threshold=4.25,\
		           null_grad_thresh=20., null_grad_val = 1./5.):
		"""
		Return the BestNR statistic for this row.
		"""
		# weight SNR by chisq
		bestnr = self.get_new_snr(index=index, nhigh=nhigh,
		                          column="chisq")
		if len(self.get_ifos()) < 3:
			return bestnr
		# recontour null SNR threshold for higher SNRs
		if self.snr > null_grad_thresh:
			null_snr_threshold += (self.snr - null_grad_thresh) * null_grad_val
		# weight SNR by null SNR
		if self.get_null_snr() > null_snr_threshold:
			bestnr /= 1 + self.get_null_snr() - null_snr_threshold
		return bestnr