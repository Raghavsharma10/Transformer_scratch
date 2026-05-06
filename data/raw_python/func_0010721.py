def split (s, delimter, trim = True, limit = 0): # pragma: no cover
		"""
		Split a string using a single-character delimter
		@params:
			`s`: the string
			`delimter`: the single-character delimter
			`trim`: whether to trim each part. Default: True
		@examples:
			```python
			ret = split("'a,b',c", ",")
			# ret == ["'a,b'", "c"]
			# ',' inside quotes will be recognized.
			```
		@returns:
			The list of substrings
		"""
		ret   = []
		special1 = ['(', ')', '[', ']', '{', '}']
		special2 = ['\'', '"']
		special3 = '\\'
		flags1 = [0, 0, 0]
		flags2 = [False, False]
		flags3 = False
		start  = 0
		nlim   = 0
		for i, c in enumerate(s):
			if c == special3:
				# next char is escaped
				flags3 = not flags3
			elif not flags3:
				# no escape
				if c in special1:
					index = special1.index(c)
					if index % 2 == 0:
						flags1[int(index/2)] += 1
					else:
						flags1[int(index/2)] -= 1
				elif c in special2:
					index = special2.index(c)
					flags2[index] = not flags2[index]
				elif c == delimter and not any(flags1) and not any(flags2):
					r = s[start:i]
					if trim: r = r.strip()
					ret.append(r)
					start = i + 1
					nlim = nlim + 1
					if limit and nlim >= limit:
						break
			else:
				# escaping closed
				flags3 = False
		r = s[start:]
		if trim: r = r.strip()
		ret.append(r)
		return ret