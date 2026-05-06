def checkin(self, package, no_partitions=False, force=False,  cb=None):
        """
        Check in a bundle package to the remote.

        :param package: A Database, referencing a sqlite database holding the bundle
        :param cb: a two argument progress callback: cb(message, num_records)
        :return:
        """
        from ambry.orm.exc import NotFoundError

        if not os.path.exists(package.path):
            raise NotFoundError("Package path does not exist: '{}' ".format(package.path))

        if self.is_api:
            return self._checkin_api(package, no_partitions=no_partitions, force=force, cb=cb)
        else:
            return self._checkin_fs(package, no_partitions=no_partitions, force=force, cb=cb)