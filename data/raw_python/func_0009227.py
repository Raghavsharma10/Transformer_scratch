def __merge_values(self, from_dict, to_dict):
        """
        Merge dictionary objects recursively, by only updating keys existing in to_dict
        """
        for key, value in from_dict.iteritems():

            # Only if the key already exists
            if key in to_dict:

                # Make sure the values are the same datatype
                assert type(to_dict[key]) is type(from_dict[key]), 'Data type for {0} is incorrect: {1}, should be {2}'.format(key, type(from_dict[key]), type(to_dict[key]))

                # Recursively dive into the next dictionary
                if type(to_dict[key]) is dict:
                    to_dict[key] = self.__merge_values(from_dict[key], to_dict[key])

                # Replace value
                else:
                    to_dict[key] = from_dict[key]

        return to_dict