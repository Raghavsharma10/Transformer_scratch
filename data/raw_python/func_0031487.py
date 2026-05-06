def get_functions(cls, entry):
        """
        get `models.Function` objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Function` objects
        """
        comments = []
        query = "./comment[@type='function']"
        for comment in entry.iterfind(query):
            text = comment.find('./text').text
            comments.append(models.Function(text=text))

        return comments