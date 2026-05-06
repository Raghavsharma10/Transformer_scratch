def alwaysThrew(self, error_type=None): #pylint: disable=invalid-name
        """
        Determining whether the specified exception is the ONLY thrown exception
        Args:
            error_type:
                None: checking without specified exception
                Specified Exception
        Return: Boolean
        """
        if self.callCount == 0:
            return False
        if not error_type:
            return True if len(self.exceptions) == self.callCount else False
        else:
            return uch.obj_in_list_always(self.exceptions, error_type)