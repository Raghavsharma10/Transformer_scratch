def outputmap(self, data):
        """ Internal function used to traverse a data structure and map the contents onto python-friendly objects inplace.

            This uses recursion, so try not to pass in anything that's over 255 objects deep.

            :param data: data structure
            :type data: any
            :param prefix: endpoint family, eg. sources, historics
            :type prefix: str
            :param endpoint: endpoint being called on the API
            :type endpoint: str
            :returns: Nothing, edits inplace
        """
        if isinstance(data, list):
            for item in data:
                self.outputmap(item)
        elif isinstance(data, dict):
            for map_target in self.output_map:
                if map_target in data:
                    data[map_target] = getattr(self, self.output_map[map_target])(data[map_target])
            for item in data.values():
                self.outputmap(item)