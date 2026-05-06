def wrap(cls, app):
        """
        Adds test live server capability to a Flask app module.
        
        :param app:
            A :class:`flask.Flask` app instance.
        """

        host, port = cls.parse_args()
        ssl = cls._argument_parser.parse_args().ssl
        ssl_context = None

        if host:
            if ssl:
                try:
                    import OpenSSL
                except ImportError:
                    # OSX fix
                    sys.path.append(
                        '/System/Library/Frameworks/Python.framework/Versions/'
                        '{0}.{1}/Extras/lib/python/'
                        .format(sys.version_info.major, sys.version_info.minor)
                    )

                try:
                    import OpenSSL
                except ImportError:
                    # Linux fix
                    sys.path.append(
                        '/usr/lib/python{0}.{1}/dist-packages/'
                        .format(sys.version_info.major, sys.version_info.minor)
                    )

                try:
                    import OpenSSL
                except ImportError:
                    raise LiveAndLetDieError(
                        'Flask app could not be launched because the pyopenssl '
                        'library is not installed on your system!'
                    )
                ssl_context = 'adhoc'

            app.run(host=host, port=port, ssl_context=ssl_context)
            sys.exit()