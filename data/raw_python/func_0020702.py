def getSrcDatasetParents(self, url, dataset):
        """
        List block at src DBS
        """
        #resturl = "%s/datasetparents?dataset=%s" % (url, dataset)
        params={'dataset':dataset}
        return cjson.decode(self.callDBSService(url, 'datasetparents', params, {}))