def getEntitiesByRegexp(data = None, listRegexp = None, verbosity=1, logFolder="./logs"):
	''' 
		Method to obtain entities by Regexp.

		:param data:	text where the entities will be looked for.
		:param listRegexp:	list of selected regular expressions to be looked for. If None was provided, all the available will be chosen instead.
		:param verbosity:	Verbosity level.
		:param logFolder:	Folder to store the logs.
		
		:return:	a list of the available objects containing the expressions found in the provided data.
		[
		  {
			"attributes": [],
			"type": "i3visio.email",
			"value": "foo@bar.com"
		  },
		  {
			"attributes": [],
			"type": "i3visio.email",
			"value": "bar@foo.com"
		  }
		]
	'''
	i3visiotools.logger.setupLogger(loggerName="entify", verbosity=verbosity, logFolder=logFolder)	
	logger = logging.getLogger("entify")
	if listRegexp == None:
		listRegexp = config.getAllRegexp()

	foundExpr = []

	for r in listRegexp:
		foundExpr += r.findExp(data)

	# print foundExpr

	return foundExpr