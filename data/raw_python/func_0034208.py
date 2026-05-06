def get_null_snr(self):
		"""
		Get the coherent Null SNR for each row in the table.
		"""
		null_snr_sq = self.get_coinc_snr()**2 - self.get_column('snr')**2
		null_snr_sq[null_snr_sq < 0] = 0.
		return null_snr_sq**(1./2.)