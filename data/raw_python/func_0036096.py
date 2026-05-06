def _genEmptyResults(self):
        """ Uses allowed keys to generate a empty dict to start counting from
        :return:
        """

        allowedKeys = self._allowedKeys

        keysDict = OrderedDict()  # Note: list comprehension take 0 then 2 then 1 then 3 etc for some reason. we want strict order
        for k in allowedKeys:
            keysDict[k] = 0


        resultsByClass = keysDict

        return resultsByClass