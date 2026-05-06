def do_pickle_ontology(filename, g=None):
	"""
	from a valid filename, generate the graph instance and pickle it too
	note: option to pass a pre-generated graph instance too
	2015-09-17: added code to increase recursion limit if cPickle fails
		see http://stackoverflow.com/questions/2134706/hitting-maximum-recursion-depth-using-pythons-pickle-cpickle
	"""
	ONTOSPY_LOCAL_MODELS = get_home_location()
	pickledpath = ONTOSPY_LOCAL_CACHE + "/" + filename + ".pickle"
	if not g:
		g = Ontospy(ONTOSPY_LOCAL_MODELS + "/" + filename)

	if not GLOBAL_DISABLE_CACHE:
		try:
			cPickle.dump(g, open(pickledpath, "wb"))
			# print Style.DIM + ".. cached <%s>" % pickledpath + Style.RESET_ALL
		except Exception as e:
			print("\n.. Failed caching <%s>" % filename )
			print(str(e))
			print("\n... attempting to increase the recursion limit from %d to %d" % (sys.getrecursionlimit(), sys.getrecursionlimit()*10))

		try:
			sys.setrecursionlimit(sys.getrecursionlimit()*10)
			cPickle.dump(g, open(pickledpath, "wb"))
			# print(Fore.GREEN + "Cached <%s>" % pickledpath + "..." + Style.RESET_ALL)
		except Exception as e:
			print("\n... Failed caching <%s>... aborting..." % filename )
			print(str(e))
		sys.setrecursionlimit(int(sys.getrecursionlimit()/10))
	return g