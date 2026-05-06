def intercept_image_formats(self, options):
        """
        Load all image formats if needed.
        """
        if 'entityTypes' in options:
            for entity in options['entityTypes']:
                if entity['type'] == ENTITY_TYPES.IMAGE and 'imageFormats' in entity:
                    if entity['imageFormats'] == '__all__':
                        entity['imageFormats'] = get_all_image_formats()

        return options