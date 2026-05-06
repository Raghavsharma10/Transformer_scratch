def _generate_consumer_tag(self):
        """Generate a unique consumer tag.

        :rtype string:

        """
        return "%s.%s%s" % (
                self.__class__.__module__,
                self.__class__.__name__,
                self._next_consumer_tag())