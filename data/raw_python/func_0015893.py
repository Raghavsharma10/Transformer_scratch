def get_vhost_names(self):
        """
        A convenience function for getting back only the vhost names instead of
        the larger vhost dicts.

        :returns list vhost_names: A list of just the vhost names.
        """
        vhosts = self.get_all_vhosts()
        vhost_names = [i['name'] for i in vhosts]
        return vhost_names