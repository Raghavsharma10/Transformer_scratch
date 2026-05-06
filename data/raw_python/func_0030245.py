def clean_value(self, value):
        '''
        Additional clean action to preprocess value before :meth:`to_python`
        method.

        Subclasses may define own clean_value method to allow additional clean
        actions like html cleanup, etc.
        '''
        # We have to clean before checking min/max length. It's done in
        # separate method to allow additional clean action in subclasses.
        if self.nontext_replacement is not None:
            value = replace_nontext(value, self.nontext_replacement)
        if self.strip:
            value = value.strip()
        return value