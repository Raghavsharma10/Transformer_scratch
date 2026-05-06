def get_keywords(self, entry):
        """
        get list of models.Keyword objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Keyword` objects
        """
        keyword_objects = []

        for keyword in entry.iterfind("./keyword"):
            identifier = keyword.get('id')
            name = keyword.text
            keyword_hash = hash(identifier)

            if keyword_hash not in self.keywords:
                self.keywords[keyword_hash] = models.Keyword(**{'identifier': identifier, 'name': name})

            keyword_objects.append(self.keywords[keyword_hash])

        return keyword_objects