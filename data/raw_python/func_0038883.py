def servicenames(self):
        "Give the list of services available in this folder."
        return set([service['name'].rstrip('/').split('/')[-1] 
                        for service in self._json_struct.get('services', [])])