def _read(self, spec_path, product_name):
        """
        Reads the spec files and extracts the concrete product spec.
        :param spec_path:
        :param product_name:
        :return:
        """
        matches = []
        with codecs.open(spec_path, 'r') as f:
            for entry in json.loads(f.read()):
                if product_name in entry.get('products'):
                    matches.append(entry)
        return matches