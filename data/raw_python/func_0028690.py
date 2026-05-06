def get_pokemon_by_name(self, name):
        """
        Returns an array of Pokemon objects containing all the forms of the
        Pokemon specified the name of the Pokemon.
        """
        endpoint = '/pokemon/' + str(name)
        return self.make_request(self.BASE_URL + endpoint)