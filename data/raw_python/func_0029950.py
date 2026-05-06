def logger(self):
        """The bundle logger."""

        if not self._logger:

            ident = self.identity

            if self.multi:
                template = '%(levelname)s %(process)d {} %(message)s'.format(ident.vid)
            else:
                template = '%(levelname)s {} %(message)s'.format(ident.vid)

            try:
                file_name = self.build_fs.getsyspath(self.log_file)
                self._logger = get_logger(__name__, template=template, stream=sys.stdout, file_name=file_name)
            except NoSysPathError:
                # file does not exists in the os - memory fs for example.
                self._logger = get_logger(__name__, template=template, stream=sys.stdout)

            self._logger.setLevel(self._log_level)

        return self._logger