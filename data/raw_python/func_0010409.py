def get_modelname(self):
        """Returns the FritzBox model name."""
        xpath = '%s/%s' % (self.nodename('device'), self.nodename('modelName'))
        return self.root.find(xpath).text