def _convert_md_type(self, type_to_convert: str):
        """Metadata types are not consistent in Isogeo API. A vector dataset is
         defined as vector-dataset in query filter but as vectorDataset in
         resource (metadata) details.

        see: https://github.com/isogeo/isogeo-api-py-minsdk/issues/29
        """
        if type_to_convert in FILTER_TYPES:
            return FILTER_TYPES.get(type_to_convert)
        elif type_to_convert in FILTER_TYPES.values():
            return [k for k, v in FILTER_TYPES.items() if v == type_to_convert][0]
        else:
            raise ValueError(
                "Incorrect metadata type to convert: {}".format(type_to_convert)
            )