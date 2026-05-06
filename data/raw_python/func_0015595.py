def _json_path_search(self, json_dict, expr):
        """
        Scan JSON dictionary with using json-path passed sting of the format of $.element..element1[index] etc.

        *Args:*\n
        _json_dict_ - JSON dictionary;\n
        _expr_ - string of fuzzy search for items within the directory;\n

        *Returns:*\n
        List of DatumInContext objects:
        ``[DatumInContext(value=..., path=..., context=[DatumInContext])]``
        - value - found value
        - path  - value selector inside context.value (in implementation of jsonpath-rw: class Index or Fields)

        *Raises:*\n
        JsonValidatorError
        """
        path = parse(expr)
        results = path.find(json_dict)

        if len(results) is 0:
            raise JsonValidatorError("Nothing found in the dictionary {0} using the given path {1}".format(
                str(json_dict), str(expr)))

        return results