def iri2uri(uri):
	"""Convert an IRI to a URI. Note that IRIs must be
	passed in a unicode strings. That is, do not utf-8 encode
	the IRI before passing it into the function."""

	assert uri != None, 'iri2uri must be passed a non-none string!'

	original = uri
	if isinstance(uri ,str):
		(scheme, authority, path, query, fragment) = urllib.parse.urlsplit(uri)
		authority = authority.encode('idna').decode('utf-8')
		# For each character in 'ucschar' or 'iprivate'
		#  1. encode as utf-8
		#  2. then %-encode each octet of that utf-8
		# path = urllib.parse.quote(path)
		uri = urllib.parse.urlunsplit((scheme, authority, path, query, fragment))
		uri = "".join([encode(c) for c in uri])

	# urllib.parse.urlunsplit(urllib.parse.urlsplit({something})
	# strips any trailing "?" chars. While this may be legal according to the
	# spec, it breaks some services. Therefore, we patch
	# the "?" back in if it has been removed.
	if original.endswith("?") and not uri.endswith("?"):
		uri = uri+"?"
	return uri