def _getPFilename(self,native,prompt):
        """Get p_filename field for this parameter

        Same as get for non-list params
        """
        return self.get(native=native,prompt=prompt)