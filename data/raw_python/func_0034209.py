def get_sigmasqs(self, instruments=None):
		"""
		Return dictionary of single-detector sigmas for each row in the
		table.
		"""
		if len(self):
			if not instruments:
				instruments = map(str, \
					instrument_set_from_ifos(self[0].ifos))
			return dict((ifo, self.get_sigmasq(ifo))\
				    for ifo in instruments)
		else:
			return dict()