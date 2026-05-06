def find_remote_bundle(self, ref, try_harder=None):
        """
        Locate a bundle, by any reference, among the configured remotes. The routine will only look in the cache
        directory lists stored in the remotes, which must be updated to be current.

        :param vid: A bundle or partition reference, vid, or name
        :param try_harder: If the reference isn't found, try parsing for an object id, or subsets of the name
        :return: (remote,vname) or (None,None) if the ref is not found
        """
        from ambry.identity import ObjectNumber

        remote, vid = self._find_remote_bundle(ref)

        if remote:
            return (remote, vid)

        if try_harder:

            on = ObjectNumber.parse(vid)

            if on:
                raise NotImplementedError()
                don = on.as_dataset
                return self._find_remote_bundle(vid)

            # Try subsets of a name, assuming it is a name
            parts = ref.split('-')

            for i in range(len(parts) - 1, 2, -1):
                remote, vid = self._find_remote_bundle('-'.join(parts[:i]))

                if remote:
                    return (remote, vid)
        return (None, None)