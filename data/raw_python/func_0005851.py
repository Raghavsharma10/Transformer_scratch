def add(self, work_dir, external=False):
        """Run a spooler on the specified directory.

        :param str|unicode work_dir:

            .. note:: Placeholders can be used to build paths, e.g.: {project_runtime_dir}/spool/
              See ``Section.project_name`` and ``Section.runtime_dir``.

        :param bool external: map spoolers requests to a spooler directory managed by an external instance

        """
        command = 'spooler'

        if external:
            command += '-external'

        self._set(command, self._section.replace_placeholders(work_dir), multi=True)

        return self._section