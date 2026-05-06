def to_json(self):
        """
        Returns the JSON representation of the editor interface.
        """

        result = super(EditorInterface, self).to_json()
        result.update({'controls': self.controls})
        return result