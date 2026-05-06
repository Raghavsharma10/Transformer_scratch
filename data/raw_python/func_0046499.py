def _serialize(self, uri, node):
        """
        Serialize node result as dict
        """
        meta = self._decode_meta(node['meta'], is_published=bool(node['is_published']))
        return {
            'uri': uri.clone(ext=node['plugin'], version=node['version']),
            'content': node['content'],
            'meta': meta
        }