def create_in_hdx(self):
        # type: () -> None
        """Check if user exists in HDX and if so, update it, otherwise create user

        Returns:
            None
        """
        capacity = self.data.get('capacity')
        if capacity is not None:
            del self.data['capacity']
        self._create_in_hdx('user', 'id', 'name')
        if capacity is not None:
            self.data['capacity'] = capacity