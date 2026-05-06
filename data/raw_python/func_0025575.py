def iiscgi(application):
	"""A specialized version of the reference WSGI-CGI server to adapt to Microsoft IIS quirks.
	
	This is not a production quality interface and will behave badly under load.
	"""
	try:
		from wsgiref.handlers import IISCGIHandler
	except ImportError:
		print("Python 3.2 or newer is required.")
	
	if not __debug__:
		warnings.warn("Interactive debugging and other persistence-based processes will not work.")
	
	IISCGIHandler().run(application)