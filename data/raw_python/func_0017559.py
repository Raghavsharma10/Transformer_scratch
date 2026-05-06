def delete_value(self, key):
        """
        Delete the key if the token is expired.

        Arg:
        key : cache key
        """
        response = {}
        response['status'] = False
        response['msg'] = "key does not exist"

        file_cache = self.read_file()
        if key in file_cache:
            del file_cache[key]
            self.update_file(file_cache)
            response['status'] = True
            response['msg'] = "success"
        return response