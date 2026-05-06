def get_change_type(self, ref, a1, a2):
        """
        Given ref, allele1, and allele2, returns the type of change.
        The only case of an amino acid insertion is when the ref is
        represented as a '.'.
        """
        if ref == '.':
            return self.INSERTION
        elif a1 == '.' or a2 == '.':
            return self.DELETION