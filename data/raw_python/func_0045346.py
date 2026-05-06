def riak_multi_get(self, key_list_tuple):
        """
        Sends given tuples of list to multiget method and took riak objs' keys and data. For each
        multiget call, separate pools are used and after execution, pools are stopped.
        Args:
            key_list_tuple(list of tuples): [('bucket_type','bucket','riak_key')]

                                            Example:
                                            [('models','personel','McAPchPZzB6RVJ8QI2XSVQk4mUR')]

        Returns:
            objs(tuple): obj's key and obj's value

        """
        pool = PyokoMG()
        objs = self._client.multiget(key_list_tuple, pool=pool)
        pool.stop()
        return objs