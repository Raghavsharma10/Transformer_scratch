def get_context_data(self):
        """
        Context Data is equal to context + extra_context
        Merge the dicts context_data and extra_context and
        update state
        """
        self.get_context()
        self.context_data.update(self.get_extra_context())
        return self.context_data