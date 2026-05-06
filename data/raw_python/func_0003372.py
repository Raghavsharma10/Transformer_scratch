def discover(self, details = False):
        'Discover API definitions. Set details=true to show details'
        if details and not (isinstance(details, str) and details.lower() == 'false'):
            return copy.deepcopy(self.discoverinfo)
        else:
            return dict((k,v.get('description', '')) for k,v in self.discoverinfo.items())