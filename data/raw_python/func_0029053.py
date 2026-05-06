def _convert_to_dictionary(obj):
        """
        Convert obj to a dictionary with formatting appropriate for a PIF. This function attempts to treat obj as
        a Pio object and otherwise returns obj.

        :param obj: Object to convert to a dictionary.
        :returns: Input object as a dictionary or the original object.
        """
        if isinstance(obj, list):
            return [Serializable._convert_to_dictionary(i) for i in obj]
        elif hasattr(obj, 'as_dictionary'):
            return obj.as_dictionary()
        else:
            return obj