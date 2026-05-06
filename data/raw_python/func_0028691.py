def get_pokemon_by_number(self, number):
        """
        Returns an array of Pokemon objects containing all the forms of the
        Pokemon specified the Pokedex number.
        """
        endpoint = '/pokemon/' + str(number)
        return self.make_request(self.BASE_URL + endpoint)