def addRnaQuantMetadata(self, fields):
        """
        data elements are:
        Id, annotations, description, name, readGroupId
        where annotations is a comma separated list
        """
        self._featureSetIds = fields["feature_set_ids"].split(',')
        self._description = fields["description"]
        self._name = fields["name"]
        self._biosampleId = fields.get("biosample_id", "")
        if fields["read_group_ids"] == "":
            self._readGroupIds = []
        else:
            self._readGroupIds = fields["read_group_ids"].split(',')
        if fields["programs"] == "":
            self._programs = []
        else:
            # Need to use program Id's here to generate a list of Programs
            # for now set to empty
            self._programs = []