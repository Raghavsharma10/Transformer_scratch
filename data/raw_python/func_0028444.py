def _version_view(self):
        """
        View that returns the contents of version.json or a 404.
        """
        version_json = self._version_callback(self.version_path)
        if version_json is None:
            return 'version.json not found', 404
        else:
            return jsonify(version_json)