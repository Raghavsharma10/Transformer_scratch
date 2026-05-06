def as_dict(self):
        """
        Serializes the object necessary data in a dictionary.

        :returns: Serialized data in a dictionary.
        :rtype: dict
        """

        result_dict = super(Profile, self).as_dict()

        statuses = list()
        version = None
        titles = list()
        descriptions = list()
        platforms = list()
        selects = list()

        for child in self.children:
            if isinstance(child, Version):
                version = child.as_dict()
            elif isinstance(child, Status):
                statuses.append(child.as_dict())
            elif isinstance(child, Title):
                titles.append(child.as_dict())
            elif isinstance(child, Description):
                descriptions.append(child.as_dict())
            elif isinstance(child, Platform):
                platforms.append(child.as_dict())
            elif isinstance(child, Select):
                selects.append(child.as_dict())

        if version is not None:
            result_dict['version'] = version
        if len(statuses) > 0:
            result_dict['statuses'] = statuses
        if len(titles) > 0:
            result_dict['titles'] = titles
        if len(descriptions) > 0:
            result_dict['descriptions'] = descriptions
        if len(platforms) > 0:
            result_dict['platforms'] = platforms
        if len(selects) > 0:
            result_dict['selects'] = selects

        return result_dict