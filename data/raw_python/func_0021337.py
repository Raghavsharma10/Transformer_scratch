def arcfour(key, csbN=1):
	'''Return a generator for the ARCFOUR/RC4 pseudorandom keystream for the 
	   key provided. Keys should be byte strings or sequences of ints.'''
	if isinstance(key, str):
		key = [ord(c) for c in key]
	s = range(256)
	j = 0
	for n in range(csbN):
		for i in range(256):
			j = (j + s[i] + key[i % len(key)]) % 256
			t = s[i]
			s[i] = s[j]
			s[j] = t
	i = 0
	j = 0
	while True:
		i = (i + 1) % 256
		j = (j + s[i]) % 256
		t = s[i] 
		s[i] = s[j]
		s[j] = t
		yield s[(s[i] + s[j]) % 256]