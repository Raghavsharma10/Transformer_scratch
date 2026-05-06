def get_evolution_stone(self, slug):
        """
        Returns a Evolution Stone object containing the details about the
        evolution stone.
        """
        endpoint = '/evolution-stone/' + slug
        return self.make_request(self.BASE_URL + endpoint)