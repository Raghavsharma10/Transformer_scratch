def _get_section_new(cls, dir_base):
        """Creates a new section with default settings.

        :param str|unicode dir_base:
        :rtype: Section

        """
        from uwsgiconf.presets.nice import PythonSection
        from django.conf import settings

        wsgi_app = settings.WSGI_APPLICATION

        path_wsgi, filename, _, = wsgi_app.split('.')
        path_wsgi = os.path.join(dir_base, path_wsgi, '%s.py' %filename)

        section = PythonSection(
            wsgi_module=path_wsgi,

        ).networking.register_socket(
            PythonSection.networking.sockets.http('127.0.0.1:8000')

        ).main_process.change_dir(dir_base)

        return section