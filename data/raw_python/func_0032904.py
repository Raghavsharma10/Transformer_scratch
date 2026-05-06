def path(self, filename=None, ext='tsv', digest=False, shard=False, encoding='utf-8'):
        """
        Return the path for this class with a certain set of parameters.
        `ext` sets the extension of the file.
        If `hash` is true, the filename (w/o extenstion) will be hashed.
        If `shard` is true, the files are placed in shards, based on the first
        two chars of the filename (hashed).
        """
        if self.BASE is NotImplemented:
            raise RuntimeError('BASE directory must be set.')

        params = dict(self.get_params())

        if filename is None:
            parts = []

            for name, param in self.get_params():
                if not param.significant:
                    continue
                if name == 'date' and is_closest_date_parameter(self, 'date'):
                    parts.append('date-%s' % self.closest())
                    continue
                if hasattr(param, 'is_list') and param.is_list:
                    es = '-'.join([str(v) for v in getattr(self, name)])
                    parts.append('%s-%s' % (name, es))
                    continue
                
                val = getattr(self, name)

                if isinstance(val, datetime.datetime):
                    val = val.strftime('%Y-%m-%dT%H%M%S')
                elif isinstance(val, datetime.date):
                    val = val.strftime('%Y-%m-%d')
                
                parts.append('%s-%s' % (name, val))

            name = '-'.join(sorted(parts))
            if len(name) == 0:
                name = 'output'
            if digest:
                name = hashlib.sha1(name.encode(encoding)).hexdigest()
            if not ext:
                filename = '{fn}'.format(ext=ext, fn=name)
            else:
                filename = '{fn}.{ext}'.format(ext=ext, fn=name)
            if shard:
                prefix = hashlib.sha1(filename.encode(encoding)).hexdigest()[:2]
                return os.path.join(self.BASE, self.TAG, self.task_family, prefix, filename)

        return os.path.join(self.BASE, self.TAG, self.task_family, filename)