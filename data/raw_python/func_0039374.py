def get_generator(self):
        """Return the generator object to anonymize data."""

        faker = Faker()

        try:
            return getattr(faker, self.category)

        except AttributeError:
            raise ValueError('Category {} couldn\'t be found on faker')