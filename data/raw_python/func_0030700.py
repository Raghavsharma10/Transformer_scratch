def siblingExtensionSearch(self, *exts):
        """Attempt to return a path with my name, given multiple possible
        extensions.

        Each extension in exts will be tested and the first path which exists
        will be returned.  If no path exists, None will be returned.  If '' is
        in exts, then if the file referred to by this path exists, 'self' will
        be returned.

        The extension '*' has a magic meaning, which means "any path that
        begins with self.path+'.' is acceptable".
        """
        p = self.path
        for ext in exts:
            if not ext and self.exists():
                return self
            if ext == '*':
                basedot = basename(p)+'.'
                for fn in listdir(dirname(p)):
                    if fn.startswith(basedot):
                        return self.clonePath(joinpath(dirname(p), fn))
            p2 = p + ext
            if exists(p2):
                return self.clonePath(p2)