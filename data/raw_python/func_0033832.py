def segmentlistdict_to_short_string(seglists):
	"""
	Return a string representation of a segmentlistdict object.  Each
	segmentlist in the dictionary is encoded using to_range_strings()
	with "," used to delimit segments.  The keys are converted to
	strings and paired with the string representations of their
	segmentlists using "=" as a delimiter.  Finally the key=value
	strings are combined using "/" to delimit them.

	Example:

	>>> from pycbc_glue.segments import *
	>>> segs = segmentlistdict({"H1": segmentlist([segment(0, 10), segment(35, 35), segment(100, infinity())]), "L1": segmentlist([segment(5, 15), segment(45, 60)])})
	>>> segmentlistdict_to_short_string(segs)
	'H1=0:10,35,100:/L1=5:15,45:60'

	This function, and its inverse segmentlistdict_from_short_string(),
	are intended to be used to allow small segmentlistdict objects to
	be encoded in command line options and config files.  For large
	segmentlistdict objects or when multiple sets of segmentlists are
	required, the LIGO Light Weight XML encoding available through the
	pycbc_glue.ligolw library should be used.
	"""
	return "/".join(["%s=%s" % (str(key), ",".join(to_range_strings(value))) for key, value in seglists.items()])