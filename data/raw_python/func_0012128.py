def transform(self, tr_list, files):
        """
        replaces $tokens$ with values
        will be replaced with config rendering
        """
        singular = False
        if not isinstance(files, list) and not isinstance(files, tuple):
            singular = True
            files = [files]

        for _find, _replace in tr_list:
            files = [opt.replace(_find, _replace) for opt in files]

        if singular:
            return files[0]
        else:
            return files