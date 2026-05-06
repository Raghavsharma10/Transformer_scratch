def scanFolderForRegexp(folder = None, listRegexp = None, recursive = False, verbosity=1, logFolder= "./logs"):
	''' 
		[Optionally] recursive method to scan the files in a given folder.

		:param folder:	the folder to be scanned.
		:param listRegexp:	listRegexp is an array of <RegexpObject>.
		:param recursive:	when True, it performs a recursive search on the subfolders.
	
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

	logger.info("Scanning the folder: " + folder)	
	results = {}

	#onlyfiles = []
	#for f in listdir(args.input_folder):
	#	if isfile(join(args.input_folder, f)):
	#		onlyfiles.append(f)	
	onlyfiles = [ f for f in listdir(folder) if isfile(join(folder,f)) ]
	
	for f in onlyfiles:
		filePath = join(folder,f)
		logger.debug("Looking for regular expressions in: " + filePath)	

		with open(filePath, "r") as tempF:
			# reading data
			foundExpr = getEntitiesByRegexp(data = tempF.read(), listRegexp = listRegexp)
			logger.debug("Updating the " + str(len(foundExpr)) + " results found on: " + filePath)	
			results[filePath] = foundExpr

	if recursive:
		onlyfolders = [ f for f in listdir(folder) if isdir(join(folder,f)) ]
		for f in onlyfolders:
			folderPath = join(folder, f)
			logger.debug("Looking for additional in the folder: "+ folderPath)
			results.update(scanFolderForRegexp(folder = folderPath,listRegexp = listRegexp, recursive = recursive))
	return results