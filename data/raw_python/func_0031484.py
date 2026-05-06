def get_recommended_protein_name(cls, entry):
        """
        get recommended full and short protein name as tuple from XML node

        :param entry: XML node entry
        :return: (str, str) => (full, short)
        """
        query_full = "./protein/recommendedName/fullName"
        full_name = entry.find(query_full).text

        short_name = None
        query_short = "./protein/recommendedName/shortName"
        short_name_tag = entry.find(query_short)
        if short_name_tag is not None:
            short_name = short_name_tag.text

        return full_name, short_name