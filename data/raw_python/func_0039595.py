def save(self, *args, **kwargs):
        """
        saves creates or updates current resource
        returns new resource
        """
        self._pre_save(*args, **kwargs)
        response = self._save(*args, **kwargs)
        response = self._post_save(response, *args, **kwargs)
        return response