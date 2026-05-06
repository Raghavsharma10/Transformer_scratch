def _translate_segment_glob(pattern):
		"""
		Translates the glob pattern to a regular expression. This is used in
		the constructor to translate a path segment glob pattern to its
		corresponding regular expression.

		*pattern* (:class:`str`) is the glob pattern.

		Returns the regular expression (:class:`str`).
		"""
		# NOTE: This is derived from `fnmatch.translate()` and is similar to
		# the POSIX function `fnmatch()` with the `FNM_PATHNAME` flag set.

		escape = False
		regex = ''
		i, end = 0, len(pattern)
		while i < end:
			# Get next character.
			char = pattern[i]
			i += 1

			if escape:
				# Escape the character.
				escape = False
				regex += re.escape(char)

			elif char == '\\':
				# Escape character, escape next character.
				escape = True

			elif char == '*':
				# Multi-character wildcard. Match any string (except slashes),
				# including an empty string.
				regex += '[^/]*'

			elif char == '?':
				# Single-character wildcard. Match any single character (except
				# a slash).
				regex += '[^/]'

			elif char == '[':
				# Braket expression wildcard. Except for the beginning
				# exclamation mark, the whole braket expression can be used
				# directly as regex but we have to find where the expression
				# ends.
				# - "[][!]" matchs ']', '[' and '!'.
				# - "[]-]" matchs ']' and '-'.
				# - "[!]a-]" matchs any character except ']', 'a' and '-'.
				j = i
				# Pass brack expression negation.
				if j < end and pattern[j] == '!':
					j += 1
				# Pass first closing braket if it is at the beginning of the
				# expression.
				if j < end and pattern[j] == ']':
					j += 1
				# Find closing braket. Stop once we reach the end or find it.
				while j < end and pattern[j] != ']':
					j += 1

				if j < end:
					# Found end of braket expression. Increment j to be one past
					# the closing braket:
					#
					#  [...]
					#   ^   ^
					#   i   j
					#
					j += 1
					expr = '['

					if pattern[i] == '!':
						# Braket expression needs to be negated.
						expr += '^'
						i += 1
					elif pattern[i] == '^':
						# POSIX declares that the regex braket expression negation
						# "[^...]" is undefined in a glob pattern. Python's
						# `fnmatch.translate()` escapes the caret ('^') as a
						# literal. To maintain consistency with undefined behavior,
						# I am escaping the '^' as well.
						expr += '\\^'
						i += 1

					# Build regex braket expression. Escape slashes so they are
					# treated as literal slashes by regex as defined by POSIX.
					expr += pattern[i:j].replace('\\', '\\\\')

					# Add regex braket expression to regex result.
					regex += expr

					# Set i to one past the closing braket.
					i = j

				else:
					# Failed to find closing braket, treat opening braket as a
					# braket literal instead of as an expression.
					regex += '\\['

			else:
				# Regular character, escape it for regex.
				regex += re.escape(char)

		return regex