def _add_nested(self, rec, name, value):
        """Adds a term's nested attributes."""
        # Remove comments and split term into typedef / target term.
        (typedef, target_term) = value.split('!')[0].rstrip().split(' ')

        # Save the nested term.
        getattr(rec, name)[typedef].append(target_term)