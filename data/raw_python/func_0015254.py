def _load_meta(self, meta):
        '''Load data from meta.yaml to a dictionary'''
        meta = yaml.load(meta, Loader=Loader)

        # Versions are often specified in a format that is convertible to an
        # int or a float, so we want to make sure it is interpreted as a str.
        # Fix for the bug #300.
        if 'version' in meta:
            meta['version'] = str(meta['version'])

        return meta