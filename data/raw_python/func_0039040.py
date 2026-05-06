def eventlog(self, path):
        """Iterates over the Events contained within the log at the given path.

        For each Event, yields a XML string.

        """
        self.logger.debug("Parsing Event log file %s.", path)

        with NamedTemporaryFile(buffering=0) as tempfile:
            self._filesystem.download(path, tempfile.name)

            file_header = FileHeader(tempfile.read(), 0)

            for xml_string, _ in evtx_file_xml_view(file_header):
                yield xml_string