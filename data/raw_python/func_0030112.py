def checkin_remote_bundle(self, ref, remote=None):
        """ Checkin a remote bundle to this library.

        :param ref: Any bundle reference
        :param remote: If specified, use this remote. If not, search for the reference
            in cached directory listings
        :param cb: A one argument progress callback
        :return:
        """

        if not remote:
            remote, vname = self.find_remote_bundle(ref)
            if vname:
                ref = vname
        else:
            pass

        if not remote:
            raise NotFoundError("Failed to find bundle ref '{}' in any remote".format(ref))

        self.logger.info("Load '{}' from '{}'".format(ref, remote))

        vid = self._checkin_remote_bundle(remote, ref)

        self.commit()

        return vid