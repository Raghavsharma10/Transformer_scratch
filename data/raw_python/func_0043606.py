def interested_in(self):
        """
        A list of strings describing the genders the user is interested in.
        """
        genders = []

        for gender in self.cache['interested_in']:
            genders.append(gender)

        return genders