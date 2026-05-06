def get_backends_versions(self):
        """
        Get backends versions
        :return: dict containing name of backend and version.
        """
        # We are not always have permission, so find open.
        for i in range(0, len(self.organizations)):
            try:
                backends = self.organizations[i].environments['default'].backends
            except ApiAuthenticationError:
                pass
            else:
                break
        versions = dict([(x['name'], x['version']) for x in backends])
        return versions