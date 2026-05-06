def from_dict(cls, d):
        """Like Identity.from_dict, but will cast the class type based on the
        format. i.e. if the format is hdf, return an HdfPartitionIdentity.

        :param d:
        :return:

        """

        name = PartitionIdentity._name_class(**d)

        if 'id' in d and 'revision' in d:
            # The vid should be constructed from the id and the revision
            on = (ObjectNumber.parse(d['id']).rev(d['revision']))
        elif 'vid' in d:
            on = ObjectNumber.parse(d['vid'])
        else:
            raise ValueError("Must have id and revision, or vid")

        try:
            return PartitionIdentity(name, on)
        except TypeError as e:
            raise TypeError(
                "Failed to make identity from \n{}\n: {}".format(
                    d,
                    e.message))