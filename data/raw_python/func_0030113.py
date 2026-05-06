def _checkin_remote_bundle(self, remote, ref):
        """
        Checkin a remote bundle from a remote
        :param remote: a Remote object
        :param ref: Any bundle reference
        :return: The vid of the loaded bundle
        """
        from ambry.bundle.process import call_interval
        from ambry.orm.exc import NotFoundError
        from ambry.orm import Remote
        from ambry.util.flo import copy_file_or_flo
        from tempfile import NamedTemporaryFile

        assert isinstance(remote, Remote)

        @call_interval(5)
        def cb(r, total):
            self.logger.info("{}: Downloaded {} bytes".format(ref, total))

        b = None
        try:
            b = self.bundle(ref)
            self.logger.info("{}: Already installed".format(ref))
            vid = b.identity.vid

        except NotFoundError:
            self.logger.info("{}: Syncing".format(ref))

            db_dir = self.filesystem.downloads('bundles')
            db_f = os.path.join(db_dir, ref) #FIXME. Could get multiple versions of same file. ie vid and vname

            if not os.path.exists(os.path.join(db_dir, db_f)):

                self.logger.info("Downloading bundle '{}' to '{}".format(ref, db_f))
                with open(db_f, 'wb') as f_out:
                    with remote.checkout(ref) as f:
                        copy_file_or_flo(f, f_out, cb=cb)
                        f_out.flush()

            self.checkin_bundle(db_f)

            b = self.bundle(ref)  # Should exist now.

            b.dataset.data['remote_name'] = remote.short_name

            b.dataset.upstream = remote.url

            b.dstate = b.STATES.CHECKEDOUT

            b.commit()

        finally:
            if b:
                b.progress.close()

        vid = b.identity.vid

        return vid