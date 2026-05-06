def instrument_set_from_ifos(ifos):
	"""
	Parse the values stored in the "ifos" and "instruments" columns
	found in many tables.  This function is mostly for internal use by
	the .instruments properties of the corresponding row classes.  The
	mapping from input to output is as follows (rules are applied in
	order):

	input is None --> output is None

	input contains "," --> output is set of strings split on "," with
	leading and trailing whitespace stripped from each piece and empty
	strings removed from the set

	input contains "+" --> output is set of strings split on "+" with
	leading and trailing whitespace stripped from each piece and empty
	strings removed from the set

	else, after stripping input of leading and trailing whitespace,

	input has an even length greater than two --> output is set of
	two-character pieces

	input is a non-empty string --> output is a set containing input as
	single value

	else output is an empty set.

	NOTE:  the complexity of this algorithm is a consequence of there
	being several conventions in use for encoding a set of instruments
	into one of these columns;  it has been proposed that L.L.W.
	documents standardize on the comma-delimited variant of the
	encodings recognized by this function, and for this reason the
	inverse function, ifos_from_instrument_set(), implements that
	encoding only.

	NOTE:  to force a string containing an even number of characters to
	be interpreted as a single instrument name and not to be be split
	into two-character pieces, add a "," or "+" character to the end to
	force the comma- or plus-delimited decoding to be used.
	ifos_from_instrument_set() does this for you.

	Example:

	>>> print instrument_set_from_ifos(None)
	None
	>>> instrument_set_from_ifos(u"")
	set([])
	>>> instrument_set_from_ifos(u"  ,  ,,")
	set([])
	>>> instrument_set_from_ifos(u"H1")
	set([u'H1'])
	>>> instrument_set_from_ifos(u"SWIFT")
	set([u'SWIFT'])
	>>> instrument_set_from_ifos(u"H1L1")
	set([u'H1', u'L1'])
	>>> instrument_set_from_ifos(u"H1L1,")
	set([u'H1L1'])
	>>> instrument_set_from_ifos(u"H1,L1")
	set([u'H1', u'L1'])
	>>> instrument_set_from_ifos(u"H1+L1")
	set([u'H1', u'L1'])
	"""
	if ifos is None:
		return None
	if u"," in ifos:
		result = set(ifo.strip() for ifo in ifos.split(u","))
		result.discard(u"")
		return result
	if u"+" in ifos:
		result = set(ifo.strip() for ifo in ifos.split(u"+"))
		result.discard(u"")
		return result
	ifos = ifos.strip()
	if len(ifos) > 2 and not len(ifos) % 2:
		# if ifos is a string with an even number of characters
		# greater than two, split it into two-character pieces.
		# FIXME:  remove this when the inspiral codes don't write
		# ifos strings like this anymore
		return set(ifos[n:n+2] for n in range(0, len(ifos), 2))
	if ifos:
		return set([ifos])
	return set()