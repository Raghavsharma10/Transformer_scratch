def _get_prefixes(self):
        """Get the possible nick prefixes and associated modes for a client."""
        prefixes = {
            "@": "o",
            "+": "v",
        }
        feature_prefixes = self.server.features.get('PREFIX')
        if feature_prefixes:
            modes = feature_prefixes[1:len(feature_prefixes)//2]
            symbols = feature_prefixes[len(feature_prefixes)//2+1:]
            prefixes = dict(zip(symbols, modes))
        return prefixes