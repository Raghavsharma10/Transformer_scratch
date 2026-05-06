def default(self):
        """Return default contents"""
        import json

        import ambry.bundle.default_files as df
        import os

        path = os.path.join(os.path.dirname(df.__file__), 'notebook.ipynb')

        if six.PY2:
            with open(path, 'rb') as f:
                content_str = f.read()
        else:
            # py3
            with open(path, 'rt', encoding='utf-8') as f:
                content_str = f.read()

        c = json.loads(content_str)

        context = {
            'title': self._bundle.metadata.about.title,
            'summary': self._bundle.metadata.about.title,
            'bundle_vname': self._bundle.identity.vname
        }

        for cell in c['cells']:
            for i in range(len(cell['source'])):
                cell['source'][i] = cell['source'][i].format(**context)

        c['metadata']['ambry'] = {
            'identity': self._bundle.identity.dict
        }

        return json.dumps(c, indent=4)