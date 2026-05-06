def glob(cls, files=None):
        '''
        Glob a pattern or a list of pattern static storage relative(s).
        '''
        files = files or []
        if isinstance(files, str):
            files = os.path.normpath(files)
            matches = lambda path: matches_patterns(path, [files])
            return [path for path in cls.get_static_files() if matches(path)]
        elif isinstance(files, (list, tuple)):
            all_files = cls.get_static_files()
            files = [os.path.normpath(f) for f in files]
            sorted_result = []
            for pattern in files:
                sorted_result.extend([f for f in all_files if matches_patterns(f, [pattern])])
            return sorted_result