def check_encoding_and_env():
        """
        It brings the environment variables to the screen.
        The user checks to see if they are using the correct variables.
        """
        import sys
        import os
        if sys.getfilesystemencoding() in ['utf-8', 'UTF-8']:
            print(__(u"{0}File system encoding correct{1}").format(CheckList.OKGREEN,
                                                                   CheckList.ENDC))
        else:
            print(__(u"{0}File system encoding wrong!!{1}").format(CheckList.FAIL,
                                                                   CheckList.ENDC))
        check_env_list = ['RIAK_PROTOCOL', 'RIAK_SERVER', 'RIAK_PORT', 'REDIS_SERVER',
                          'DEFAULT_BUCKET_TYPE', 'PYOKO_SETTINGS',
                          'MQ_HOST', 'MQ_PORT', 'MQ_USER', 'MQ_VHOST',
                          ]
        env = os.environ
        for k, v in env.items():
            if k in check_env_list:
                print(__(u"{0}{1} : {2}{3}").format(CheckList.BOLD, k, v, CheckList.ENDC))