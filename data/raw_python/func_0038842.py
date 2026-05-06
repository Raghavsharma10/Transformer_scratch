def _get_translated_field_names(model_instance):
        """
        Get the instance translatable fields

        :return:
        """
        hvad_internal_fields = ['id', 'language_code', 'master', 'master_id', 'master_id']
        translated_field_names = set(model_instance._translated_field_names) - set(hvad_internal_fields)
        return translated_field_names