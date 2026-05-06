def ifos_from_instrument_set(instruments):
	"""
	Convert an iterable of instrument names into a value suitable for
	storage in the "ifos" column found in many tables.  This function
	is mostly for internal use by the .instruments properties of the
	corresponding row classes.  The input can be None or an iterable of
	zero or more instrument names, none of which may be zero-length,
	consist exclusively of spaces, or contain "," or "+" characters.
	The output is a single string containing the unique instrument
	names concatenated using "," as a delimiter.  instruments will only
	be iterated over once and so can be a generator expression.
	Whitespace is allowed in instrument names but might not be
	preserved.  Repeated names will not be preserved.

	NOTE:  in the special case that there is 1 instrument name in the
	iterable and it has an even number of characters > 2 in it, the
	output will have a "," appended in order to force
	instrument_set_from_ifos() to parse the string back into a single
	instrument name.  This is a special case included temporarily to
	disambiguate the encoding until all codes have been ported to the
	comma-delimited encoding.  This behaviour will be discontinued at
	that time.  DO NOT WRITE CODE THAT RELIES ON THIS!  You have been
	warned.

	Example:

	>>> print ifos_from_instrument_set(None)
	None
	>>> ifos_from_instrument_set(())
	u''
	>>> ifos_from_instrument_set((u"H1",))
	u'H1'
	>>> ifos_from_instrument_set((u"H1",u"H1",u"H1"))
	u'H1'
	>>> ifos_from_instrument_set((u"H1",u"L1"))
	u'H1,L1'
	>>> ifos_from_instrument_set((u"SWIFT",))
	u'SWIFT'
	>>> ifos_from_instrument_set((u"H1L1",))
	u'H1L1,'
	"""
	if instruments is None:
		return None
	_instruments = sorted(set(instrument.strip() for instrument in instruments))
	# safety check:  refuse to accept blank names, or names with commas
	# or pluses in them as they cannot survive the encode/decode
	# process
	if not all(_instruments) or any(u"," in instrument or u"+" in instrument for instrument in _instruments):
		raise ValueError(instruments)
	if len(_instruments) == 1 and len(_instruments[0]) > 2 and not len(_instruments[0]) % 2:
		# special case disambiguation.  FIXME:  remove when
		# everything uses the comma-delimited encoding
		return u"%s," % _instruments[0]
	return u",".join(_instruments)