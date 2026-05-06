def check_riak():
        """
            Riak checks the connection
            It displays on the screen whether or not you have a connection.
        """
        from pyoko.db.connection import client
        from socket import error as socket_error

        try:
            if client.ping():
                print(__(u"{0}Riak is working{1}").format(CheckList.OKGREEN, CheckList.ENDC))
            else:
                print(__(u"{0}Riak is not working{1}").format(CheckList.FAIL, CheckList.ENDC))
        except socket_error as e:
            print(__(u"{0}Riak is not working{1}").format(CheckList.FAIL,
                                                          CheckList.ENDC), e.message)