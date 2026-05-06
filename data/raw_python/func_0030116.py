def _find_remote_bundle(self, ref, remote_service_type='s3'):
        """
        Locate a bundle, by any reference, among the configured remotes. The routine will
        only look in the cache directory lists stored in the remotes, which must
        be updated to be current.

        :param ref:
        :return: (remote,vname) or (None,None) if the ref is not found
        """

        for r in self.remotes:

            if remote_service_type and r.service != remote_service_type:
                continue

            if 'list' not in r.data:
                continue

            for k, v in r.data['list'].items():
                if ref in v.values():
                    return (r, v['vname'])

        return None, None