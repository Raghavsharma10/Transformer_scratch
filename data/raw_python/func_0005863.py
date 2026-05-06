def configure():
    """Configure uWSGI.

    This returns several configuration objects, which will be used
    to spawn several uWSGI processes.

    Applications are on 127.0.0.1 on ports starting from 8000.

    """
    import os
    from uwsgiconf.presets.nice import PythonSection

    FILE = os.path.abspath(__file__)
    port = 8000

    configurations = []

    for idx in range(2):

        alias = 'app_%s' % (idx + 1)

        section = PythonSection(
            # Automatically reload uWSGI if this file is changed.
            touch_reload=FILE,

            # To differentiate easily.
            process_prefix=alias,

            # Serve WSGI application (see above) from this very file.
            wsgi_module=FILE,

            # Custom WSGI callable for second app.
            wsgi_callable=alias,

            # One is just enough, no use in worker on every core
            # for this demo.
            workers=1,

        ).networking.register_socket(
            PythonSection.networking.sockets.http('127.0.0.1:%s' % port)
        )

        port += 1

        configurations.append(
            # We give alias for configuration to prevent clashes.
            section.as_configuration(alias=alias))

    return configurations