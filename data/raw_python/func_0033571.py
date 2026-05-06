def get_resource(self):
        '''
        Returns a BytesResource to build the viewers JavaScript
        '''
        # Basename could be used for controlling caching
        # basename = 'viewers_%s' % settings.get_cache_string()
        node_packages = self.get_node_packages()
        # sort_keys is essential to ensure resulting string is
        # deterministic (and thus hashable)
        viewers_data_str = json.dumps(node_packages, sort_keys=True)
        viewers_data = viewers_data_str.encode('utf8')
        viewers_resource = ForeignBytesResource(
            viewers_data,
            extension=VIEWER_EXT,
            # basename=basename,
        )
        return viewers_resource