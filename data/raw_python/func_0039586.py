def post_process(self, paths, dry_run=False, **options):
        """
        Overridden to allow some files to be excluded (using
        ``postprocess_exclusions``)

        """
        if self.postprocess_exclusions:
            paths = dict((k, v) for k, v in paths.items() if not
                    self.exclude_file(k))
        return super(CachedFilesMixin, self).post_process(paths,
                dry_run, **options)