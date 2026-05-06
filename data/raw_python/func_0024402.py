def from_string(cls, alg_str):
        """
        Creates a location from a two character string consisting of 
        the file then rank written in algebraic notation.
        
        Examples: e4, b5, a7

        :type: alg_str: str
        :rtype: Location
        """
        try:
            return cls(int(alg_str[1]) - 1, ord(alg_str[0]) - 97)
        except ValueError as e:
            raise ValueError("Location.from_string {} invalid: {}".format(alg_str, e))