def cluster_version(self):
        """Get version of Elasticsearch running on the cluster."""
        versionstr = self.client.info()['version']['number']
        return [int(x) for x in versionstr.split('.')]