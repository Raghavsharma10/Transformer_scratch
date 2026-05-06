def get_annotation(self, key, result_format='list'):
        """
        Is a convenience method for accessing annotations on models that have them
        """
        value = self.get('_annotations_by_key', {}).get(key)
        if not value:
            return value

        if result_format == 'one':
            return value[0]

        return value