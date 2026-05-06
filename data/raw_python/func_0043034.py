def load(self, name):
        """
        If not yet in the cache, load the named template and compiles it,
        placing it into the cache.

        If in cache, return the cached template.
        """

        if self.reload:
            self._maybe_purge_cache()

        template = self.cache.get(name)
        if template:
            return template

        path = self.resolve(name)
        if not path:
            raise OSError(errno.ENOENT, "File not found: %s" % name)

        with codecs.open(path, 'r', encoding='UTF-8') as f:
            contents = f.read()
            mtime = os.fstat(f.fileno()).st_mtime

        template = self.load_string(contents, filename=path)
        template.mtime = mtime
        template.path = path

        self.cache[name] = template
        return template