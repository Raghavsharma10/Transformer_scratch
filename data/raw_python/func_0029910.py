def parse(cls, on_str, force_type=None):  # @ReservedAssignment
        """Parse a string into one of the object number classes."""

        on_str_orig = on_str

        if on_str is None:
            return None

        if not on_str:
            raise NotObjectNumberError("Got null input")

        if not isinstance(on_str, string_types):
            raise NotObjectNumberError("Must be a string. Got a {} ".format(type(on_str)))

        # if isinstance(on_str, unicode):
        #     dataset = on_str.encode('ascii')

        if force_type:
            type_ = force_type
        else:
            type_ = on_str[0]

        on_str = on_str[1:]

        if type_ not in list(cls.NDS_LENGTH.keys()):
            raise NotObjectNumberError("Unknown type character '{}' for '{}'".format(type_, on_str_orig))

        ds_length = len(on_str) - cls.NDS_LENGTH[type_]

        if ds_length not in cls.DATASET_LENGTHS:
            raise NotObjectNumberError(
                "Dataset string '{}' has an unfamiliar length: {}".format(on_str_orig, ds_length))

        ds_lengths = cls.DATASET_LENGTHS[ds_length]

        assignment_class = ds_lengths[2]

        try:
            dataset = int(ObjectNumber.base62_decode(on_str[0:ds_lengths[0]]))

            if ds_lengths[1]:
                i = len(on_str) - ds_lengths[1]
                revision = int(ObjectNumber.base62_decode(on_str[i:]))
                on_str = on_str[0:i]  # remove the revision
            else:
                revision = None

            on_str = on_str[ds_lengths[0]:]

            if type_ == cls.TYPE.DATASET:
                return DatasetNumber(dataset, revision=revision, assignment_class=assignment_class)

            elif type_ == cls.TYPE.TABLE:
                table = int(ObjectNumber.base62_decode(on_str))
                return TableNumber(
                    DatasetNumber(dataset, assignment_class=assignment_class), table, revision=revision)

            elif type_ == cls.TYPE.PARTITION:
                partition = int(ObjectNumber.base62_decode(on_str))
                return PartitionNumber(
                    DatasetNumber(dataset, assignment_class=assignment_class), partition, revision=revision)

            elif type_ == cls.TYPE.COLUMN:
                table = int(ObjectNumber.base62_decode(on_str[0:cls.DLEN.TABLE]))
                column = int(ObjectNumber.base62_decode(on_str[cls.DLEN.TABLE:]))

                return ColumnNumber(
                    TableNumber(DatasetNumber(dataset, assignment_class=assignment_class), table),
                    column, revision=revision)

            elif type_ == cls.TYPE.OTHER1 or type_ == cls.TYPE.CONFIG:
                    return GeneralNumber1(on_str_orig[0],
                                          DatasetNumber(dataset, assignment_class=assignment_class),
                                          int(ObjectNumber.base62_decode(on_str[0:cls.DLEN.OTHER1])),
                                          revision=revision)

            elif type_ == cls.TYPE.OTHER2:
                    return GeneralNumber2(on_str_orig[0],
                                          DatasetNumber(dataset, assignment_class=assignment_class),
                                          int(ObjectNumber.base62_decode(on_str[0:cls.DLEN.OTHER1])),
                                          int(ObjectNumber.base62_decode(
                                              on_str[cls.DLEN.OTHER1:cls.DLEN.OTHER1+cls.DLEN.OTHER2])),
                                          revision=revision)

            else:

                raise NotObjectNumberError('Unknown type character: ' + type_ + ' in ' + str(on_str_orig))

        except Base62DecodeError as e:
            raise NotObjectNumberError('Unknown character:  ' + str(e))