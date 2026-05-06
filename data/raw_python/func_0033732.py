def __set_whitelist(self, whitelist=None):
        """
        Update default white list by customer white list
        """
        # add tag's names as key and list of enabled attributes as value for defaults
        self.whitelist = {}
        # tags that removed with contents
        self.sanitizelist = ['script', 'style']
        if isinstance(whitelist, dict) and '*' in whitelist.keys():
            self.isNotPurify = True
            self.whitelist_keys = []
            return
        else:
            self.isNotPurify = False
        self.whitelist.update(whitelist or {})
        self.whitelist_keys = self.whitelist.keys()