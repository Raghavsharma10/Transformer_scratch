def collect(self):
        """
        Perform the bulk of the work of collectstatic.

        Split off from handle_noargs() to facilitate testing.
        """
        if self.symlink:
            if sys.platform == 'win32':
                raise CommandError("Symlinking is not supported by this "
                                   "platform (%s)." % sys.platform)
            if not self.local:
                raise CommandError("Can't symlink to a remote destination.")

        if self.clear:
            self.clear_dir('')

        handler = self._get_handler()

        do_post_process = self.post_process and hasattr(self.storage, 'post_process')

        found_files = SortedDict()
        for finder in finders.get_finders():
            for path, storage in finder.list(self.ignore_patterns):
                # Prefix the relative path if the source storage contains it
                if getattr(storage, 'prefix', None):
                    prefixed_path = os.path.join(storage.prefix, path)
                else:
                    prefixed_path = path

                if prefixed_path not in found_files:
                    found_files[prefixed_path] = (storage, path)
                    handler(path, prefixed_path, storage)
                    if self.progressive_post_process and do_post_process:
                        try:
                            self._post_process(
                                    {prefixed_path: (storage, path)},
                                    self.dry_run)
                        except ValueError as e:
                            message = ('%s current storage requires all files'
                                ' to have been collected first. Try '
                                ' ecstatic.storage.CachedStaticFilesStorage' \
                                % e)
                            raise ValueError(message)

        if not self.progressive_post_process and do_post_process:
            self._post_process(found_files, self.dry_run)

        return {
            'modified': self.copied_files + self.symlinked_files,
            'unmodified': self.unmodified_files,
            'post_processed': self.post_processed_files,
        }