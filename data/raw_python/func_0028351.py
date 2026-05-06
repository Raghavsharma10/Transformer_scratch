def get_errors(self):
        """If there were any business errors fetching data for this property,
        returns the error messages.

        Returns:
            string - the error message, or None if there was no error.

        """
        return [{cr.component_name: cr.get_error()}
                for cr in self.component_results if cr.has_error()]