def as_dict(self):
        """
        Serializes the object necessary data in a dictionary.

        :returns: Serialized data in a dictionary.
        :rtype: dict
        """

        result_dict = super(Benchmark, self).as_dict()

        statuses = list()
        titles = list()
        descriptions = list()
        front_matters = list()
        rear_matters = list()
        platforms = list()
        version = None
        profiles = list()
        groups = list()

        for child in self.children:
            if isinstance(child, Version):
                version = child.as_dict()
            elif isinstance(child, Status):
                statuses.append(child.as_dict())
            elif isinstance(child, Title):
                titles.append(child.as_dict())
            elif isinstance(child, Description):
                descriptions.append(child.as_dict())
            elif isinstance(child, FrontMatter):
                front_matters.append(child.as_dict())
            elif isinstance(child, RearMatter):
                rear_matters.append(child.as_dict())
            elif isinstance(child, Platform):
                platforms.append(child.as_dict())
            elif isinstance(child, Profile):
                profiles.append(child.as_dict())
            elif isinstance(child, Group):
                groups.append(child.as_dict())

        if version is not None:
            result_dict['version'] = version
        if len(statuses) > 0:
            result_dict['statuses'] = statuses
        if len(titles) > 0:
            result_dict['titles'] = titles
        if len(descriptions) > 0:
            result_dict['descriptions'] = descriptions
        if len(front_matters) > 0:
            result_dict['front_matters'] = front_matters
        if len(rear_matters) > 0:
            result_dict['rear_matters'] = rear_matters
        if len(platforms) > 0:
            result_dict['platforms'] = platforms
        if len(profiles) > 0:
            result_dict['profiles'] = profiles
        if len(groups) > 0:
            result_dict['groups'] = groups

        return result_dict