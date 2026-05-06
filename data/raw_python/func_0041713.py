def __get_preferential_value(self, paths, index=0):
        """
        Returns the preferential path's value. Preferential path being the
        first keyword (a.k.a. path) found in the path order list created on
        instantiation.
        """

        try:
            value = paths[self.path_preference_order[index]]
        except KeyError:
            value = self.__get_preferential_value(paths, (index + 1))
        except IndexError:
            msg = ('Cannot fork to any of the provided path\'s values. '
                   'Perhaps add a fallback path (set to `True`) in your '
                   'fork\'s instantiation?')
            raise self.PathNotAvailable(msg)

        return value