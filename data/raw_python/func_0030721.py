def geocode(self):
        """A Generator that reads from the address generators and returns
        geocode results.

        The generator yields ( address, geocode_results, object)

        """

        submit_set = []
        data_map = {}

        for address, o in self.gen:
            submit_set.append(address)
            data_map[address] = o

            if len(submit_set) >= self.submit_size:
                results = self._send(submit_set)
                submit_set = []

                for k, result in results.items():
                    o = data_map[k]
                    yield (k, result, o)

        if len(submit_set) > 0:
            results = self._send(submit_set)
            # submit_set = []

            for k, result in results.items():
                o = data_map[k]
                yield (k, result, o)