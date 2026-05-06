def handle_entityref(self, name):
        """
        Handler of processing entity (overrided, private)
        """
        self.log.debug( u'Encountered entity  : {0}'.format(name) )
        if not self.removeEntity:
            self.data.append('&%s;' % name)