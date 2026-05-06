def fromlalcache(cachefile, coltype = int):
	"""
	Construct a segmentlist representing the times spanned by the files
	identified in the LAL cache contained in the file object file.  The
	segmentlist will be created with segments whose boundaries are of
	type coltype, which should raise ValueError if it cannot convert
	its string argument.

	Example:

	>>> from pycbc_glue.lal import LIGOTimeGPS
	>>> cache_seglists = fromlalcache(open(filename), coltype = LIGOTimeGPS).coalesce()

	See also:

	pycbc_glue.lal.CacheEntry
	"""
	return segments.segmentlist(lal.CacheEntry(l, coltype = coltype).segment for l in cachefile)