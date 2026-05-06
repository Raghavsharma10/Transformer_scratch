def unregister(self, bucket, name):
        """
        Remove the function from the registry by name
        """
        assert bucket in self, 'Bucket %s is unknown' % bucket
        if not name in self[bucket]:
            raise NotRegistered('The function %s is not registered' % name)
        del self[bucket][name]