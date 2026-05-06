def mirror_sources(self, sourcedir, targetdir=None, recursive=True,
                       excludes=[]):
        """
        Mirroring compilable sources filepaths to their targets.

        Args:
            sourcedir (str): Directory path to scan.

        Keyword Arguments:
            absolute (bool): Returned paths will be absolute using
                ``sourcedir`` argument (if True), else return relative paths.
            recursive (bool): Switch to enabled recursive finding (if True).
                Default to True.
            excludes (list): A list of excluding patterns (glob patterns).
                Patterns are matched against the relative filepath (from its
                sourcedir).

        Returns:
            list: A list of pairs ``(source, target)``. Where ``target`` is the
                ``source`` path but renamed with ``.css`` extension. Relative
                directory from source dir is left unchanged but if given,
                returned paths will be absolute (using ``sourcedir`` for
                sources and ``targetdir`` for targets).
        """
        sources = self.compilable_sources(
            sourcedir,
            absolute=False,
            recursive=recursive,
            excludes=excludes
        )
        maplist = []

        for filepath in sources:
            src = filepath
            dst = self.get_destination(src, targetdir=targetdir)

            # In absolute mode
            if targetdir:
                src = os.path.join(sourcedir, src)

            maplist.append((src, dst))

        return maplist