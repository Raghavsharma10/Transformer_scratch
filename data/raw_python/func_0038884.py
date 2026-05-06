def services(self):
        "Returns a list of Service objects available in this folder"
        return [self._get_subfolder("%s/%s/" % 
                (s['name'].rstrip('/').split('/')[-1], s['type']), 
                self._service_type_mapping.get(s['type'], Service)) for s
                in self._json_struct.get('services', [])]