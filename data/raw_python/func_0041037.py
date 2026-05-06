def entify_main(args):
	''' 
		Main function. This function is created in this way so as to let other applications make use of the full configuration capabilities of the application.	
	'''
	# Recovering the logger
	# Calling the logger when being imported
	i3visiotools.logger.setupLogger(loggerName="entify", verbosity=args.verbose, logFolder=args.logfolder)	
	# From now on, the logger can be recovered like this:
	logger = logging.getLogger("entify")

	logger.info("""entify-launcher.py Copyright (C) F. Brezo and Y. Rubio (i3visio) 2014
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain conditions.
For details, run:
\tpython entify-launcher.py --license""")

	logger.info("Selecting the regular expressions to be analysed...")

	listRegexp = []
	if args.regexp:
		listRegexp = config.getRegexpsByName(args.regexp)

	elif args.new_regexp:
		for i, r in enumerate(args.new_regexp):
			list.Regexp.append(RegexpObject(name = "NewRegexp"+str(i), reg_exp = args.new_regexp))

	if not args.web:
		results = scanFolderForRegexp(folder = args.input_folder, listRegexp= listRegexp, recursive = args.recursive, verbosity=args.verbose, logFolder= args.logfolder)
	else:
		results = scanResource(uri = args.web, listRegexp= listRegexp, verbosity=args.verbose, logFolder= args.logfolder)
	logger.info("Printing the results:\n" + general.dictToJson(results))

	if args.output_folder:
		logger.info("Preparing the output folder...")
		if not os.path.exists(args.output_folder):
			logger.warning("The output folder \'" + args.output_folder + "\' does not exist. The system will try to create it.")
			os.makedirs(args.output_folder)
		logger.info("Storing the results...")
		"""if "csv" in args.extension:
			with open(os.path.join(args.output_folder, "results.csv"), "w") as oF:
				oF.write(resultsToCSV(results))"""
		if "json" in args.extension:
			with open(os.path.join(args.output_folder, "results.json"), "w") as oF:
				oF.write(general.dictToJson(results))