def _get_formatted_data(cls, path, context=None, site=None, language=None):
        """ Return an object to conveniently access the appropriate values. """
        return FormattedMetadata(cls(), cls._get_instances(path, context, site, language), path, site, language)