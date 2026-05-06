def generic_meta_translator(self, meta_to_translate):
        '''Translates the metadate contained in an object into a dictionary
        suitable for export.

        Parameters
        ----------
        meta_to_translate : Meta
            The metadata object to translate

        Returns
        -------
        dict
            A dictionary of the metadata for each variable of an output file
            e.g. netcdf4'''
        export_dict = {}
        if self._meta_translation_table is not None:
            # Create a translation table for the actual values of the meta labels.
            # The instrument specific translation table only stores the names of the
            # attributes that hold the various meta labels
            translation_table = {}
            for key in self._meta_translation_table:
                translation_table[getattr(self, key)] = self._meta_translation_table[key]
        else:
            translation_table = None
        #First Order Data
        for key in meta_to_translate.data.index:
            if translation_table is None:
                export_dict[key] = meta_to_translate.data.loc[key].to_dict()
            else:
                # Translate each key if a translation is provided
                export_dict[key] = {}
                meta_dict = meta_to_translate.data.loc[key].to_dict()
                for original_key in meta_dict:
                    if original_key in translation_table:
                        for translated_key in translation_table[original_key]:
                            export_dict[key][translated_key] = meta_dict[original_key]
                    else:
                        export_dict[key][original_key] = meta_dict[original_key]


        #Higher Order Data
        for key in meta_to_translate.ho_data:
            if key not in export_dict:
                export_dict[key] = {}
            for ho_key in meta_to_translate.ho_data[key].data.index:
                if translation_table is None:
                    export_dict[key+'_'+ho_key] = meta_to_translate.ho_data[key].data.loc[ho_key].to_dict()
                else:
                    #Translate each key if a translation is provided
                    export_dict[key+'_'+ho_key] = {}
                    meta_dict = meta_to_translate.ho_data[key].data.loc[ho_key].to_dict()
                    for original_key in meta_dict:
                        if original_key in translation_table:
                            for translated_key in translation_table[original_key]:
                                export_dict[key+'_'+ho_key][translated_key] = meta_dict[original_key]
                        else:
                            export_dict[key+'_'+ho_key][original_key] = meta_dict[original_key]

        return export_dict