def get_banks(self):
        """Gets the bank list resulting from a search.

        return: (osid.assessment.BankList) - the bank list
        raise:  IllegalState - the bank list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.BankList(self._results, runtime=self._runtime)