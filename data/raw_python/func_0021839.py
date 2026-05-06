def print_download_uri(self, version, source):
        """
        @param version: version number or 'dev' for svn
        @type version: string

        @param source: download source or egg
        @type source: boolean

        @returns: None

        """

        if version == "dev":
            pkg_type = "subversion"
            source = True
        elif source:
            pkg_type = "source"
        else:
            pkg_type = "egg"

        #Use setuptools monkey-patch to grab url
        url = get_download_uri(self.project_name, version, source,
                self.options.pypi_index)
        if url:
            print("%s" % url)
        else:
            self.logger.info("No download URL found for %s" % pkg_type)