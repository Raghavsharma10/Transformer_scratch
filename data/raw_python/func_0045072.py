def rename_pickled_ontology(filename, newname):
	""" try to rename a cached ontology """
	pickledfile = ONTOSPY_LOCAL_CACHE + "/" + filename + ".pickle"
	newpickledfile = ONTOSPY_LOCAL_CACHE + "/" + newname + ".pickle"
	if os.path.isfile(pickledfile) and not GLOBAL_DISABLE_CACHE:
		os.rename(pickledfile, newpickledfile)
		return True
	else:
		return None