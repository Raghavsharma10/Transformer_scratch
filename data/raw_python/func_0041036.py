def scanResource(uri = None, listRegexp = None, verbosity=1, logFolder= "./logs"):
	''' 
		[Optionally] recursive method to scan the files in a given folder.

		:param uri:	the URI to be scanned.
		:param listRegexp:	listRegexp is an array of <RegexpObject>.

		:return:	a dictionary where the key is the name of the file.
	'''
	i3visiotools.logger.setupLogger(loggerName="entify", verbosity=verbosity, logFolder=logFolder)
	logger = logging.getLogger("entify")

	results = {}

	logger.debug("Looking for regular expressions in: " + uri)	
	
	import urllib2
	
	foundExpr = getEntitiesByRegexp(data = urllib2.urlopen(uri).read(), listRegexp = listRegexp)
	logger.debug("Updating the " + str(len(foundExpr)) + " results found on: " + uri)	
	results[uri] = foundExpr

	return results