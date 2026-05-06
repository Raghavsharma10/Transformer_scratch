def get_response(self):
        '''
        Returns response according submitted the data and method.
        '''
        self.process_commmon()
        self.process_data()
        urlencoded_data = urllib.urlencode(self.data)
        if self.METHOD == POST:
            req = urllib2.Request(self.URL, urlencoded_data)
        else:
            req = urllib2.Request('%s?%s' %(self.URL, urlencoded_data))

        if not self.data['content']:
            raise PasteException("No content to paste")

        self.response = urllib2.urlopen(req)
        return self.response