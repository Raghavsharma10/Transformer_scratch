def _maybe_update(self, user, attribute, new_value):
        """
        DRY helper.  If the specified attribute of the user differs from the
        specified value, it will be updated.
        """
        old_value = getattr(user, attribute)
        if new_value != old_value:
            self.stderr.write(
                _('Setting {attribute} for user "{username}" to "{new_value}"').format(
                    attribute=attribute, username=user.username, new_value=new_value
                )
            )
            setattr(user, attribute, new_value)