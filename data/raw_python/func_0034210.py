def get_sngl_snrs(self, instruments=None):
		"""
		Get the single-detector SNRs for each row in the table.
		"""
		if len(self) and instruments is None:
			instruments = map(str, \
			                instrument_set_from_ifos(self[0].ifos))
		elif instruments is None:
			instruments = []
		return dict((ifo, self.get_sngl_snr(ifo))\
		            for ifo in instruments)