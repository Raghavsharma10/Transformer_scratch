def update_in_hdx(self):
        # type: () -> None
        """Check if user exists in HDX and if so, update user

        Returns:
            None
        """
        capacity = self.data.get('capacity')
        if capacity is not None:
            del self.data['capacity']  # remove capacity (which comes from users from Organization)
        self._update_in_hdx('user', 'id')
        if capacity is not None:
            self.data['capacity'] = capacity