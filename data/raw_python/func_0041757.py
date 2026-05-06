def getObservers(self):
        """
        Get the list of observer to the instance of the class.

        :return: Subscribed Obversers.
        :rtype: Array
        """
        result = []
        for observer in self._observers:
            result.append(
                          {
                              "observing": observer["observing"],
                              "call": observer["call"]
                          })
        return result