def check_redis():
        """
            Redis checks the connection
            It displays on the screen whether or not you have a connection.
        """
        from pyoko.db.connection import cache
        from redis.exceptions import ConnectionError

        try:
            cache.ping()
            print(CheckList.OKGREEN + "{0}Redis is working{1}" + CheckList.ENDC)
        except ConnectionError as e:
            print(__(u"{0}Redis is not working{1} ").format(CheckList.FAIL,
                                                            CheckList.ENDC), e.message)