def getSrcBlockParents(self, url, block):
        """
        List block at src DBS
        """
        #blockname = block.replace("#", urllib.quote_plus('#'))
        #resturl = "%s/blockparents?block_name=%s" % (url, blockname)
        params={'block_name':block}
        return cjson.decode(self.callDBSService(url, 'blockparents', params, {}))