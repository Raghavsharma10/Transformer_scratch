def get_or_create_home_repo(reset=False):
	"""
	Check to make sure we never operate with a non-existing local repo
	"""
	dosetup = True
	if os.path.exists(ONTOSPY_LOCAL):
		dosetup = False

		if reset:
			import shutil
			var = input("Delete the local library and all of its contents? (y/n) ")
			if var == "y":
				shutil.rmtree(ONTOSPY_LOCAL)
				dosetup = True
			else:
				pass

	if dosetup or not(os.path.exists(ONTOSPY_LOCAL)):
		os.mkdir(ONTOSPY_LOCAL)
	if dosetup or not(os.path.exists(ONTOSPY_LOCAL_CACHE)):
		# print "HERE"
		os.makedirs(ONTOSPY_LOCAL_CACHE)
	if dosetup or not(os.path.exists(ONTOSPY_LOCAL_VIZ)):
		os.mkdir(ONTOSPY_LOCAL_VIZ)
	if dosetup or not(os.path.exists(ONTOSPY_LIBRARY_DEFAULT)):
		os.mkdir(ONTOSPY_LIBRARY_DEFAULT)

	LIBRARY_HOME = get_home_location()  # from init file, or default

	# check that the local library folder exists, otherwise prompt user to create it
	if not(os.path.exists(LIBRARY_HOME)):
		printDebug("Warning: the local library at '%s' has been deleted or is not accessible anymore." % LIBRARY_HOME, "important")
		printDebug("Please reset the local library by running 'ontospy-manager -u <a-valid-path>'", "comment")
		raise SystemExit(1)

	if dosetup:
		print(Fore.GREEN + "Setup successfull: local library created at <%s>" % LIBRARY_HOME + Style.RESET_ALL)
	# else:
		# print(Style.DIM + "Local library: <%s>" % LIBRARY_HOME + Style.RESET_ALL)

	return True