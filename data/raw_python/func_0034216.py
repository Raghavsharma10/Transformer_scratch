def get_sngl_snrs(self):
		"""
		Return a dictionary of single-detector SNRs for this row.
		"""
		return dict((ifo, self.get_sngl_snr(ifo)) for ifo in\
                            instrument_set_from_ifos(self.ifos))