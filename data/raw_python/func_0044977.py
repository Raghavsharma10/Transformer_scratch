def delete_if_exists(self, **kwargs):
        """
        Deletes an object if it exists in database according to given query 
        parameters and returns True otherwise does nothing and returns False.
        
        Args:
            **kwargs: query parameters

        Returns(bool): True or False 

        """
        try:
            self.get(**kwargs).blocking_delete()
            return True
        except ObjectDoesNotExist:
            return False