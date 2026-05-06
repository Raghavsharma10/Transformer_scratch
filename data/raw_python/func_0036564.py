def _get_section_values(self, config, section):
        """ extract src and dst values from a section
        """
        src_host = self._get_hosts_from_names(config.get(section, 'src.host')) \
            if config.has_option(section, 'src.host') else None
        src_file = [self._get_abs_filepath(config.get(section, 'src.file'))] \
            if config.has_option(section, 'src.file') else None
        if src_host is None and src_file is None:
            raise conferr('Section "%s" gets no sources' % section)

        dst_host = self._get_hosts_from_names(config.get(section, 'dst.host')) \
            if config.has_option(section, 'dst.host') else None
        dst_file = [self._get_abs_filepath(config.get(section, 'dst.file'))] \
            if config.has_option(section, 'dst.file') else None
        if dst_host is None and dst_file is None:
            raise conferr('Section "%s" gets no destinations' % section)

        return (src_host, src_file, dst_host, dst_file)