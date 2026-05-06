def cache_train(self):
        """
        Loads the data for this classifier from a cache file

        :return: whether or not we were successful
        :rtype: bool
        """
        filename = self.get_cache_location()

        if not os.path.exists(filename):
            return False

        categories = pickle.load(open(filename, 'rb'))

        assert isinstance(categories, BayesCategories), \
            "Cache data is either corrupt or invalid"

        self.categories = categories

        # Updating our per-category overall probabilities
        self.calculate_category_probability()

        return True