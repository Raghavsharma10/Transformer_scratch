def release(self):
        """
        Release the lock if acquired.
        """
        # TODO: thread safety (currently the lock may be acquired for one more TTL length)
        if self.token is not None:
            try:
                self.client.test_and_set(self.key, 0, self.token)
            except (ValueError, etcd.EtcdKeyError, etcd.EtcdKeyNotFound) as e:
                pass  # the key already expired or got acquired by someone else
            finally:
                self.token = None