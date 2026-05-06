def _iter_tree_next(root_full, dir_rel, memo, on_error, follow_links):
	"""
	Scan the directory for all descendant files.

	*root_full* (:class:`str`) the absolute path to the root directory.

	*dir_rel* (:class:`str`) the path to the directory to scan relative to
	*root_full*.

	*memo* (:class:`dict`) keeps track of ancestor directories
	encountered. Maps each ancestor real path (:class:`str``) to relative
	path (:class:`str`).

	*on_error* (:class:`~collections.abc.Callable` or :data:`None`)
	optionally is the error handler for file-system exceptions.

	*follow_links* (:class:`bool`) is whether to walk symbolik links that
	resolve to directories.
	"""
	dir_full = os.path.join(root_full, dir_rel)
	dir_real = os.path.realpath(dir_full)

	# Remember each encountered ancestor directory and its canonical
	# (real) path. If a canonical path is encountered more than once,
	# recursion has occurred.
	if dir_real not in memo:
		memo[dir_real] = dir_rel
	else:
		raise RecursionError(real_path=dir_real, first_path=memo[dir_real], second_path=dir_rel)

	for node in os.listdir(dir_full):
		node_rel = os.path.join(dir_rel, node)
		node_full = os.path.join(root_full, node_rel)

		# Inspect child node.
		try:
			node_stat = os.lstat(node_full)
		except OSError as e:
			if on_error is not None:
				on_error(e)
			continue

		if stat.S_ISLNK(node_stat.st_mode):
			# Child node is a link, inspect the target node.
			is_link = True
			try:
				node_stat = os.stat(node_full)
			except OSError as e:
				if on_error is not None:
					on_error(e)
				continue
		else:
			is_link = False

		if stat.S_ISDIR(node_stat.st_mode) and (follow_links or not is_link):
			# Child node is a directory, recurse into it and yield its
			# decendant files.
			for file_rel in _iter_tree_next(root_full, node_rel, memo, on_error, follow_links):
				yield file_rel

		elif stat.S_ISREG(node_stat.st_mode):
			# Child node is a file, yield it.
			yield node_rel

	# NOTE: Make sure to remove the canonical (real) path of the directory
	# from the ancestors memo once we are done with it. This allows the
	# same directory to appear multiple times. If this is not done, the
	# second occurance of the directory will be incorrectly interpreted as
	# a recursion. See <https://github.com/cpburnz/python-path-specification/pull/7>.
	del memo[dir_real]