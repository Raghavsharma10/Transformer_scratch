def arcfour_drop(key, n=3072):
	'''Return a generator for the RC4-drop pseudorandom keystream given by 
	   the key and number of bytes to drop passed as arguments. Dropped bytes
	   default to the more conservative 3072, NOT the SCAN default of 768.'''
	af = arcfour(key)
	[af.next() for c in range(n)]
	return af