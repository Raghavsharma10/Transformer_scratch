def fetch(self):
        """
        Download a package

        @returns: 0 = success or 1 if failed download

        """
        #Default type to download
        source = True
        directory = "."

        if self.options.file_type == "svn":
            version = "dev"
            svn_uri = get_download_uri(self.project_name, \
                    "dev", True)
            if svn_uri:
                directory = self.project_name + "_svn"
                return self.fetch_svn(svn_uri, directory)
            else:
                self.logger.error(\
                    "ERROR: No subversion repository found for %s" % \
                    self.project_name)
                return 1
        elif self.options.file_type == "source":
            source = True
        elif self.options.file_type == "egg":
            source = False

        uri = get_download_uri(self.project_name, self.version, source)
        if uri:
            return self.fetch_uri(directory, uri)
        else:
            self.logger.error("No %s URI found for package: %s " % \
                    (self.options.file_type, self.project_name))
            return 1