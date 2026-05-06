def threw(self, error_type=None):
        """
        Determining whether the exception is thrown
        Args:
            error_type:
                None: checking without specified exception
                Specified Exception
        Return: Boolean
        """
        if not error_type:
            return True if len(self.exceptions) > 0 else False
        else:
            return uch.obj_in_list(self.exceptions, error_type)