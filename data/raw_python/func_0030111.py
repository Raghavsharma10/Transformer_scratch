def send_to_remote(self, b, no_partitions=False):
        """
        Copy a bundle to a new Sqlite file, then store the file on the remote.

        :param b: The bundle
        :return:
        """

        raise DeprecationWarning("Don't use any more?")

        from ambry.bundle.process import call_interval

        remote_name = self.resolve_remote(b)

        remote = self.remote(remote_name)

        db_path = b.package()

        with b.progress.start('checkin', 0, message='Check in bundle') as ps:

            ps.add(message='Checking in bundle {} to {}'.format(b.identity.vname, remote))

            db_ck = b.identity.cache_key + '.db'

            ps.add(message='Upload bundle file', item_type='bytes', item_count=0)
            total = [0]

            @call_interval(5)
            def upload_cb(n):
                total[0] += n
                ps.update(message='Upload bundle file', item_count=total[0])

            with open(db_path) as f:
                remote.makedir(os.path.dirname(db_ck), recursive=True, allow_recreate=True)
                self.logger.info('Send bundle file {} '.format(db_path))
                e = remote.setcontents_async(db_ck, f, progress_callback=upload_cb)
                e.wait()

            ps.update(state='done')

            if not no_partitions:
                for p in b.partitions:

                    ps.add(message='Upload partition', item_type='bytes', item_count=0, p_vid=p.vid)

                    with p.datafile.open(mode='rb') as fin:

                        total = [0]

                        @call_interval(5)
                        def progress(bytes):
                            total[0] += bytes
                            ps.update(
                                message='Upload partition'.format(p.identity.vname),
                                item_count=total[0])

                        remote.makedir(os.path.dirname(p.datafile.path), recursive=True, allow_recreate=True)
                        event = remote.setcontents_async(p.datafile.path, fin, progress_callback=progress)
                        event.wait()

                        ps.update(state='done')

            ps.add(message='Setting metadata')
            ident = json.dumps(b.identity.dict)
            remote.setcontents(os.path.join('_meta', 'vid', b.identity.vid), ident)
            remote.setcontents(os.path.join('_meta', 'id', b.identity.id_), ident)
            remote.setcontents(os.path.join('_meta', 'vname', text_type(b.identity.vname)), ident)
            remote.setcontents(os.path.join('_meta', 'name', text_type(b.identity.name)), ident)
            ps.update(state='done')

            b.dataset.commit()

            return remote_name, db_ck