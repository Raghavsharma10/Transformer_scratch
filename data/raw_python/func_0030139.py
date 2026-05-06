def localize(self, ps=None):
        """Copy a non-local partition file to the local build directory"""
        from filelock import FileLock
        from ambry.util import ensure_dir_exists
        from ambry_sources import MPRowsFile
        from fs.errors import ResourceNotFoundError

        if self.is_local:
            return

        local = self._bundle.build_fs

        b = self._bundle.library.bundle(self.identity.as_dataset().vid)

        remote = self._bundle.library.remote(b)

        lock_path = local.getsyspath(self.cache_key + '.lock')

        ensure_dir_exists(lock_path)

        lock = FileLock(lock_path)

        if ps:
            ps.add_update(message='Localizing {}'.format(self.identity.name),
                          partition=self,
                          item_type='bytes',
                          state='downloading')

        if ps:
            def progress(bts):
                if ps.rec.item_total is None:
                    ps.rec.item_count = 0

                if not ps.rec.data:
                    ps.rec.data = {}  # Should not need to do this.
                    return self

                item_count = ps.rec.item_count + bts
                ps.rec.data['updates'] = ps.rec.data.get('updates', 0) + 1

                if ps.rec.data['updates'] % 32 == 1:
                    ps.update(message='Localizing {}'.format(self.identity.name),
                              item_count=item_count)
        else:
            from ambry.bundle.process import call_interval
            @call_interval(5)
            def progress(bts):
                self._bundle.log("Localizing {}. {} bytes downloaded".format(self.vname, bts))

        def exception_cb(e):
            raise e

        with lock:
            # FIXME! This won't work with remote ( http) API, only FS ( s3:, file:)

            if self.is_local:
                return self

            try:
                with remote.fs.open(self.cache_key + MPRowsFile.EXTENSION, 'rb') as f:
                    event = local.setcontents_async(self.cache_key + MPRowsFile.EXTENSION,
                                                    f,
                                                    progress_callback=progress,
                                                    error_callback=exception_cb)
                    event.wait()
                    if ps:
                        ps.update_done()
            except ResourceNotFoundError as e:
                from ambry.orm.exc import NotFoundError
                raise NotFoundError("Failed to get MPRfile '{}' from {}: {} "
                                    .format(self.cache_key, remote.fs, e))

        return self