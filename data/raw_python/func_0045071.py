def del_pickled_ontology(filename):
	""" try to remove a cached ontology """
	pickledfile = ONTOSPY_LOCAL_CACHE + "/" + filename + ".pickle"
	if os.path.isfile(pickledfile) and not GLOBAL_DISABLE_CACHE:
		os.remove(pickledfile)
		return True
	else:
		return None