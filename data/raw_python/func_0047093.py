def removeHtmlTags(self, text):
		"""convert bad tags into HTML identities"""
		sb = []
		text = self.removeHtmlComments(text)
		bits = text.split(u'<')
		sb.append(bits.pop(0))
		tagstack = []
		tablestack = tagstack
		for x in bits:
			m = _tagPattern.match(x)
			if not m:
				continue
			slash, t, params, brace, rest = m.groups()
			t = t.lower()
			badtag = False
			if t in _htmlelements:
				# Check our stack
				if slash:
					# Closing a tag...
					if t in _htmlsingleonly or len(tagstack) == 0:
						badtag = True
					else:
						ot = tagstack.pop()
						if ot != t:
							if ot in _htmlsingleallowed:
								# Pop all elements with an optional close tag
								# and see if we find a match below them
								optstack = []
								optstack.append(ot)
								while True:
									if len(tagstack) == 0:
										break
									ot = tagstack.pop()
									if ot == t or ot not in _htmlsingleallowed:
										break
									optstack.append(ot)
								if t != ot:
									# No match. Push the optinal elements back again
									badtag = True
									tagstack += reversed(optstack)
							else:
								tagstack.append(ot)
								# <li> can be nested in <ul> or <ol>, skip those cases:
								if ot not in _htmllist and t in _listtags:
									badtag = True
						elif t == u'table':
							if len(tablestack) == 0:
								bagtag = True
							else:
								tagstack = tablestack.pop()
					newparams = u''
				else:
					# Keep track for later
					if t in _tabletags and u'table' not in tagstack:
						badtag = True
					elif t in tagstack and t not in _htmlnest:
						badtag = True
					# Is it a self-closed htmlpair? (bug 5487)
					elif brace == u'/>' and t in _htmlpairs:
						badTag = True
					elif t in _htmlsingleonly:
						# Hack to force empty tag for uncloseable elements
						brace = u'/>'
					elif t in _htmlsingle:
						# Hack to not close $htmlsingle tags
						brace = None
					else:
						if t == u'table':
							tablestack.append(tagstack)
							tagstack = []
						tagstack.append(t)
					newparams = self.fixTagAttributes(params, t)
				if not badtag:
					rest = rest.replace(u'>', u'&gt;')
					if brace == u'/>':
						close = u' /'
					else:
						close = u''
					sb.append(u'<')
					sb.append(slash)
					sb.append(t)
					sb.append(newparams)
					sb.append(close)
					sb.append(u'>')
					sb.append(rest)
					continue
			sb.append(u'&lt;')
			sb.append(x.replace(u'>', u'&gt;'))

		# Close off any remaining tags
		while tagstack:
			t = tagstack.pop()
			sb.append(u'</')
			sb.append(t)
			sb.append(u'>\n')
			if t == u'table':
				if not tablestack:
					break
				tagstack = tablestack.pop()

		return u''.join(sb)