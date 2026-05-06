def pattern_to_regex(cls, pattern):
		"""
		Convert the pattern into a regular expression.

		*pattern* (:class:`unicode` or :class:`bytes`) is the pattern to
		convert into a regular expression.

		Returns the uncompiled regular expression (:class:`unicode`, :class:`bytes`,
		or :data:`None`), and whether matched files should be included
		(:data:`True`), excluded (:data:`False`), or if it is a
		null-operation (:data:`None`).
		"""
		if isinstance(pattern, unicode):
			return_type = unicode
		elif isinstance(pattern, bytes):
			return_type = bytes
			pattern = pattern.decode(_BYTES_ENCODING)
		else:
			raise TypeError("pattern:{!r} is not a unicode or byte string.".format(pattern))

		pattern = pattern.strip()

		if pattern.startswith('#'):
			# A pattern starting with a hash ('#') serves as a comment
			# (neither includes nor excludes files). Escape the hash with a
			# back-slash to match a literal hash (i.e., '\#').
			regex = None
			include = None

		elif pattern == '/':
			# EDGE CASE: According to `git check-ignore` (v2.4.1), a single
			# '/' does not match any file.
			regex = None
			include = None

		elif pattern:

			if pattern.startswith('!'):
				# A pattern starting with an exclamation mark ('!') negates the
				# pattern (exclude instead of include). Escape the exclamation
				# mark with a back-slash to match a literal exclamation mark
				# (i.e., '\!').
				include = False
				# Remove leading exclamation mark.
				pattern = pattern[1:]
			else:
				include = True

			if pattern.startswith('\\'):
				# Remove leading back-slash escape for escaped hash ('#') or
				# exclamation mark ('!').
				pattern = pattern[1:]

			# Split pattern into segments.
			pattern_segs = pattern.split('/')

			# Normalize pattern to make processing easier.

			if not pattern_segs[0]:
				# A pattern beginning with a slash ('/') will only match paths
				# directly on the root directory instead of any descendant
				# paths. So, remove empty first segment to make pattern relative
				# to root.
				del pattern_segs[0]

			elif len(pattern_segs) == 1 or (len(pattern_segs) == 2 and not pattern_segs[1]):
				# A single pattern without a beginning slash ('/') will match
				# any descendant path. This is equivalent to "**/{pattern}". So,
				# prepend with double-asterisks to make pattern relative to
				# root.
				# EDGE CASE: This also holds for a single pattern with a
				# trailing slash (e.g. dir/).
				if pattern_segs[0] != '**':
					pattern_segs.insert(0, '**')

			else:
				# EDGE CASE: A pattern without a beginning slash ('/') but
				# contains at least one prepended directory (e.g.
				# "dir/{pattern}") should not match "**/dir/{pattern}",
				# according to `git check-ignore` (v2.4.1).
				pass

			if not pattern_segs[-1] and len(pattern_segs) > 1:
				# A pattern ending with a slash ('/') will match all descendant
				# paths if it is a directory but not if it is a regular file.
				# This is equivilent to "{pattern}/**". So, set last segment to
				# double asterisks to include all descendants.
				pattern_segs[-1] = '**'

			# Build regular expression from pattern.
			output = ['^']
			need_slash = False
			end = len(pattern_segs) - 1
			for i, seg in enumerate(pattern_segs):
				if seg == '**':
					if i == 0 and i == end:
						# A pattern consisting solely of double-asterisks ('**')
						# will match every path.
						output.append('.+')
					elif i == 0:
						# A normalized pattern beginning with double-asterisks
						# ('**') will match any leading path segments.
						output.append('(?:.+/)?')
						need_slash = False
					elif i == end:
						# A normalized pattern ending with double-asterisks ('**')
						# will match any trailing path segments.
						output.append('/.*')
					else:
						# A pattern with inner double-asterisks ('**') will match
						# multiple (or zero) inner path segments.
						output.append('(?:/.+)?')
						need_slash = True
				elif seg == '*':
					# Match single path segment.
					if need_slash:
						output.append('/')
					output.append('[^/]+')
					need_slash = True
				else:
					# Match segment glob pattern.
					if need_slash:
						output.append('/')
					output.append(cls._translate_segment_glob(seg))
					if i == end and include is True:
						# A pattern ending without a slash ('/') will match a file
						# or a directory (with paths underneath it). E.g., "foo"
						# matches "foo", "foo/bar", "foo/bar/baz", etc.
						# EDGE CASE: However, this does not hold for exclusion cases
						# according to `git check-ignore` (v2.4.1).
						output.append('(?:/.*)?')
					need_slash = True
			output.append('$')
			regex = ''.join(output)

		else:
			# A blank pattern is a null-operation (neither includes nor
			# excludes files).
			regex = None
			include = None

		if regex is not None and return_type is bytes:
			regex = regex.encode(_BYTES_ENCODING)

		return regex, include