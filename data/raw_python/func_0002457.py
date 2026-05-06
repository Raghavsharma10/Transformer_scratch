def _create_attach_record(self, id, timed):
        """
        Create a new pivot attachement record.
        """
        record = super(MorphToMany, self)._create_attach_record(id, timed)

        record[self._morph_type] = self._morph_class

        return record