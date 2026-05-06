def join(self, fm_new, minimal_subset=True):
        """
        Adds content of a new Datamat to this Datamat.

        If a parameter of the Datamats is not equal or does not exist
        in one, it is promoted to a field.

        If the two Datamats have different fields then the elements for the
        Datamats that did not have the field will be NaN, unless
        'minimal_subset' is true, in which case the mismatching fields will
        simply be deleted.

        Parameters
        fm_new : instance of Datamat
            This Datamat is added to the current one.
        minimal_subset : if true, remove fields which don't exist in both,
        	instead of using NaNs for missing elements (defaults to False)

        Capacity to use superset of fields added by rmuil 2012/01/30

        """
        # Check if parameters are equal. If not, promote them to fields.
        '''
        for (nm, val) in fm_new._parameters.items():
            if self._parameters.has_key(nm):
                if (val != self._parameters[nm]):
                    self.parameter_to_field(nm)
                    fm_new.parameter_to_field(nm)
            else:
                fm_new.parameter_to_field(nm)
        '''
        # Deal with mismatch in the fields
        # First those in self that do not exist in new...
        orig_fields = self._fields[:]
        for field in orig_fields:
            if not field in fm_new._fields:
                if minimal_subset:
                    self.rm_field(field)
                else:
                    warnings.warn("This option is deprecated. Clean and Filter your data before it is joined.", DeprecationWarning)
                    fm_new.add_field_like(field, self.field(field))
        # ... then those in the new that do not exist in self.
        orig_fields = fm_new._fields[:]
        for field in orig_fields:
            if not field in self._fields:
                if minimal_subset:
                    fm_new.rm_field(field)
                else:
                    warnings.warn("This option is deprecated. Clean and Filter your data before it is joined.", DeprecationWarning)
                    self.add_field_like(field, fm_new.field(field))

        if 'SUBJECTINDEX' in self._fields[:]:
            if fm_new.SUBJECTINDEX[0] in self.SUBJECTINDEX:
                fm_new.SUBJECTINDEX[:] = self.SUBJECTINDEX.max()+1
        # Concatenate fields
        for field in self._fields:
            self.__dict__[field] = ma.hstack((self.__dict__[field],
                fm_new.__dict__[field]))

        # Update _num_fix
        self._num_fix += fm_new._num_fix