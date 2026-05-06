def _get_file_by_value(cls, value):
        """Look up a file DataObject by name, uuid, and/or md5.
        """
        # Ignore any FileResource with no DataObject. This is a typical state
        # for a deleted file that has not yet been cleaned up.
        queryset = FileResource.objects.exclude(data_object__isnull=True)
        matches = FileResource.filter_by_name_or_id_or_tag_or_hash(
            value, queryset=queryset)
        if matches.count() == 0:
            raise ValidationError(
                'No file found that matches value "%s"' % value)
        elif matches.count() > 1:
            match_id_list = ['%s@%s' % (match.filename, match.get_uuid())
                             for match in matches]
            match_id_string = ('", "'.join(match_id_list))
            raise ValidationError(
                'Multiple files were found matching value "%s": "%s". '\
                'Use a more precise identifier to select just one file.' % (
                    value, match_id_string))
        return matches.first().data_object