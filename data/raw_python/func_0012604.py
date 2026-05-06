def annotate (self, src_dm, data_field, key_field, take_first=True):
        """
        Adds a new field (data_field) to the Datamat with data from the
        corresponding field of another Datamat (src_dm).

				This is accomplished through the use of a key_field, which is
        used to determine how the data is copied.

				This operation corresponds loosely to an SQL join operation.

        The two Datamats are essentially aligned by the unique values
        of key_field so that each block element of the new field of the target
        Datamat will consist of those elements of src_dm's data_field
        where the corresponding element in key_field matches.

        If 'take_first' is not true, and there is not
        only a single corresponding element (typical usage case) then the
        target element value will be
        a sequence (array) of all the matching elements.

        The target Datamat (self) must not have a field name data_field
        already, and both Datamats must have key_field.

        The new field in the target Datamat will be a masked array to handle
        non-existent data.

				TODO: Make example more generic, remove interoceptive reference
				TODO: Make standalone test
        Examples:

        >>> dm_intero = load_interoception_files ('test-ecg.csv', silent=True)
        >>> dm_emotiv = load_emotivestimuli_files ('test-bpm.csv', silent=True)
        >>> length(dm_intero)
        4
        >>> unique(dm_intero.subject_id)
        ['p05', 'p06']
        >>> length(dm_emotiv)
        3
        >>> unique(dm_emotiv.subject_id)
        ['p04', 'p05', 'p06']
        >>> 'interospective_awareness' in dm_intero.fieldnames()
        True
        >>> unique(dm_intero.interospective_awareness) == [0.5555, 0.6666]
        True
        >>> 'interospective_awareness' in dm_emotiv.fieldnames()
        False
        >>> dm_emotiv.copy_field(dm_intero, 'interospective_awareness', 'subject_id')
        >>> 'interospective_awareness' in dm_emotiv.fieldnames()
        True
        >>> unique(dm_emotiv.interospective_awareness) == [NaN, 0.5555, 0.6666]
        False
        """
        if key_field not in self._fields or key_field not in src_dm._fields:
            raise AttributeError('key field (%s) must exist in both Datamats'%(key_field))
        if data_field not in src_dm._fields:
            raise AttributeError('data field (%s) must exist in source Datamat' % (data_field))
        if data_field in self._fields:
            raise AttributeError('data field (%s) already exists in target Datamat' % (data_field))

        #Create a mapping of key_field value to data value.
        data_to_copy = dict([(x.field(key_field)[0], x.field(data_field)) for x in src_dm.by_field(key_field)])

        data_element = list(data_to_copy.values())[0]

        #Create the new data array of correct size.
        # We use a masked array because it is possible that for some elements
        # of the target Datamat, there exist simply no data in the source
        # Datamat. NaNs are fine as indication of this for floats, but if the
        # field happens to hold booleans or integers or something else, NaN
        # does not work.
        new_shape = [len(self)] + list(data_element.shape)
        new_data = ma.empty(new_shape, data_element.dtype)
        new_data.mask=True
        if np.issubdtype(new_data.dtype, np.float):
            new_data.fill(np.NaN) #For backwards compatibility, if mask not used

        #Now we copy the data. If the data to copy contains only a single value,
        # it is added to the target as a scalar (single value).
        # Otherwise, it is copied as is, i.e. as a sequence.
        for (key, val) in list(data_to_copy.items()):
            if take_first:
                new_data[self.field(key_field) == key] = val[0]
            else:
                new_data[self.field(key_field) == key] = val

        self.add_field(data_field, new_data)