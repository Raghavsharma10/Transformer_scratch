def get_current_response(self):
        """
        reads the current response data from the object and returns
        it in a dict.

        Currently 'time' is reported as 0 until clock drift issues are
        resolved.
        """
        response = {'port': 0,
                    'pressed': False,
                    'key': 0,
                    'time': 0}
        if len(self.__response_structs_queue) > 0:
            # make a copy just in case any other internal members of
            # XidConnection were tracking the structure
            response = self.__response_structs_queue[0].copy()
            # we will now hand over 'response' to the calling code,
            # so remove it from the internal queue
            self.__response_structs_queue.pop(0)

        return response