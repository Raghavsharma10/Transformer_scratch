def match_tree(self, root, on_error=None, follow_links=None):
		"""
		Walks the specified root path for all files and matches them to this
		path-spec.

		*root* (:class:`str`) is the root directory to search for files.

		*on_error* (:class:`~collections.abc.Callable` or :data:`None`)
		optionally is the error handler for file-system exceptions. See
		:func:`~pathspec.util.iter_tree` for more information.


		*follow_links* (:class:`bool` or :data:`None`) optionally is whether
		to walk symbolik links that resolve to directories. See
		:func:`~pathspec.util.iter_tree` for more information.

		Returns the matched files (:class:`~collections.abc.Iterable` of
		:class:`str`).
		"""
		files = util.iter_tree(root, on_error=on_error, follow_links=follow_links)
		return self.match_files(files)