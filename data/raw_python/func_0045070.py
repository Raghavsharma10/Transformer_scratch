def get_pickled_ontology(filename):
	""" try to retrieve a cached ontology """
	pickledfile = ONTOSPY_LOCAL_CACHE + "/" + filename + ".pickle"
	if GLOBAL_DISABLE_CACHE:
		printDebug("WARNING: DEMO MODE cache has been disabled in __init__.py ==============", "red")
	if os.path.isfile(pickledfile) and not GLOBAL_DISABLE_CACHE:
		try:
			return cPickle.load(open(pickledfile, "rb"))
		except:
			print("** WARNING: Cache is out of date ** ...recreating it... ")
			return None
	else:
		return None