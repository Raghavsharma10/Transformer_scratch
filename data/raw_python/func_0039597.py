def delete(self,  *args, **kwargs):
        """
        deletes current resource
        returns response from api
        """
        self._pre_delete(*args, **kwargs)
        response = self._delete(*args, **kwargs)
        response = self._post_delete(response, *args, **kwargs)
        return response