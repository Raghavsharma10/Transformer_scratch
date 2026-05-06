def _get_flatchoices(self):
        """
        Redefine standard method.

        Return constants themselves instead of their names for right rendering
        in admin's 'change_list' view, if field is present in 'list_display'
        attribute of model's admin.
        """
        return [
            (self.to_python(choice), value) for choice, value in self._choices
        ]