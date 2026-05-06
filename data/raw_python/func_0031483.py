def get_features(cls, entry):
        """
        get list of `models.Feature` from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Feature`
        """
        features = []

        for feature in entry.iterfind("./feature"):

            feature_dict = {
                'description': feature.attrib.get('description'),
                'type_': feature.attrib['type'],
                'identifier': feature.attrib.get('id')
            }

            features.append(models.Feature(**feature_dict))

        return features