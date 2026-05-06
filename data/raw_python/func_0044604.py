def match_conditions(self, filepath, sourcedir=None, nopartial=True,
                         exclude_patterns=[], excluded_libdirs=[]):
        """
        Find if a filepath match all required conditions.

        Available conditions are (in order):

        * Is allowed file extension;
        * Is a partial source;
        * Is from an excluded directory;
        * Is matching an exclude pattern;

        Args:
            filepath (str): Absolute filepath to match against conditions.

        Keyword Arguments:
            sourcedir (str or None): Absolute sources directory path. Can be
                ``None`` but then the exclude_patterns won't be matched against
                (because this method require to distinguish source dir from lib
                dirs).
            nopartial (bool): Accept partial sources if ``False``. Default is
                ``True`` (partial sources fail matchind condition). See
                ``Finder.is_partial()``.
            exclude_patterns (list): List of glob patterns, if filepath match
                one these pattern, it wont match conditions. See
                ``Finder.is_allowed()``.
            excluded_libdirs (list): A list of directory to match against
                filepath, if filepath starts with one them, it won't
                match condtions.

        Returns:
            bool: ``True`` if match all conditions, else ``False``.
        """
        # Ensure libdirs ends with / to avoid missmatching with
        # 'startswith' usage
        excluded_libdirs = [os.path.join(d, "") for d in excluded_libdirs]

        # Match an filename extension admitted as compilable stylesheet
        filename, ext = os.path.splitext(filepath)
        ext = ext[1:]
        if ext not in self.FINDER_STYLESHEET_EXTS:
            return False

        # Not a partial source
        if nopartial and self.is_partial(filepath):
            return False

        # Not in an excluded directory
        if any(
            filepath.startswith(excluded_path)
            for excluded_path in paths_by_depth(excluded_libdirs)
        ):
            return False

        # Not matching an exclude pattern
        if sourcedir and exclude_patterns:
            candidates = [sourcedir]+excluded_libdirs
            relative_path = self.get_relative_from_paths(filepath, candidates)
            if not self.is_allowed(relative_path, excludes=exclude_patterns):
                return False

        return True