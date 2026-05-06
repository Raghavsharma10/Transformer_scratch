def get(self, arg):
        """
        Return instance object with given EC2 ID or nametag.
        """
        try:
            reservations = self.get_all_instances(filters={'tag:Name': [arg]})
            instance = reservations[0].instances[0]
        except IndexError:
            try:
                instance = self.get_all_instances([arg])[0].instances[0]
            except (_ResponseError, IndexError):
                # TODO: encapsulate actual exception for debugging
                err = "Can't find any instance with name or ID '%s'" % arg
                raise ValueError(err)
        return instance