def paths(self):
        """All the entries of sys.path, possibly restricted by --path"""
        if not self.select_paths:
            return sys.path
        result = []
        match_any = set()
        for path in sys.path:
            path = os.path.normcase(os.path.abspath(path))
            for match in self.select_paths:
                match = os.path.normcase(os.path.abspath(match))
                if '*' in match:
                    if re.search(fnmatch.translate(match+'*'), path):
                        result.append(path)
                        match_any.add(match)
                        break
                else:
                    if path.startswith(match):
                        result.append(path)
                        match_any.add(match)
                        break
            else:
                logger.debug("Skipping path %s because it doesn't match %s"
                             % (path, ', '.join(self.select_paths)))
        for match in self.select_paths:
            if match not in match_any and '*' not in match:
                result.append(match)
                logger.debug("Adding path %s because it doesn't match anything already on sys.path"
                             % match)
        return result