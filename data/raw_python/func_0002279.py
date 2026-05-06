def get_title(self):
        """
        Return the string literal that is used in the template.
        The title is used in the admin screens.
        """
        try:
            return extract_literal(self.meta_kwargs['title'])
        except KeyError:
            slot = self.get_slot()
            if slot is not None:
                return slot.replace('_', ' ').title()

            return None