def handle_data(self, data):
        """
        Handler of processing data inside tag (overrided, private)
        """
        self.log.debug( u'Encountered some data  : {0}'.format(data) )
        if not self.level:
            self.data.append(data)