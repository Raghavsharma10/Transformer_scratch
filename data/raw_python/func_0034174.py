def local_path_from_url(url):
	"""
	For URLs that point to locations in the local filesystem, extract
	and return the filesystem path of the object to which they point.
	As a special case pass-through, if the URL is None, the return
	value is None.  Raises ValueError if the URL is not None and does
	not point to a local file.

	Example:

	>>> print local_path_from_url(None)
	None
	>>> local_path_from_url("file:///home/me/somefile.xml.gz")
	'/home/me/somefile.xml.gz'
	"""
	if url is None:
		return None
	scheme, host, path = urlparse.urlparse(url)[:3]
	if scheme.lower() not in ("", "file") or host.lower() not in ("", "localhost"):
		raise ValueError("%s is not a local file" % repr(url))
	return path