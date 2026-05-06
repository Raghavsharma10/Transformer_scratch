def get_string(self, key):
        """Get a string from all_options"""
        if key not in self.all_options:
            kwargs = {}
            if len(self.chain) > 1:
                kwargs['source'] = Meta(self.all_options, self.chain[-2]).source
            raise BadOptionFormat("Can't find key in options", key=key, chain=self.chain, **kwargs)

        # Make sure we special case the "content" option
        if type(key) is str and key.startswith("content."):
            return self.no_format(self.all_options["content"][key[8:]])
        if type(key) is list and len(key) is 2 and key[0] == "content":
            return self.no_format(self.all_options[key])

        return super(MergedOptionStringFormatter, self).get_string(key)