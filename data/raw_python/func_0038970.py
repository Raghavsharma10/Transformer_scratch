def keys_with_value(dictionary, value):
        "Returns a subset of keys from the dict with the value supplied."

        subset = [key for key in dictionary if dictionary[key] == value]

        return subset