def file_finder(dirname="."):
	"""
	Find the files in ``dirname`` under Mercurial version control
	according to the setuptools spec (see
	http://peak.telecommunity.com/DevCenter/setuptools#adding-support-for-other-revision-control-systems
	).
	"""
	import distutils.log
	dirname = dirname or '.'
	try:
		valid_mgrs = managers.RepoManager.get_valid_managers(dirname)
		valid_mgrs = managers.RepoManager.existing_only(valid_mgrs)
		for mgr in valid_mgrs:
			try:
				return mgr.find_all_files()
			except Exception:
				e = sys.exc_info()[1]
				distutils.log.warn(
					"hgtools.%s could not find files: %s",
					mgr, e)
	except Exception:
		e = sys.exc_info()[1]
		distutils.log.warn(
			"Unexpected error finding valid managers in "
			"hgtools.file_finder_plugin: %s", e)
	return []