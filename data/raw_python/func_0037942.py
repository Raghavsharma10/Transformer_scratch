def get_valid_managers(cls, location):
		"""
		Get the valid RepoManagers for this location.
		"""
		def by_priority_attr(c):
			return getattr(c, 'priority', 0)
		classes = sorted(
			iter_subclasses(cls), key=by_priority_attr,
			reverse=True)
		all_managers = (c(location) for c in classes)
		return (mgr for mgr in all_managers if mgr.is_valid())