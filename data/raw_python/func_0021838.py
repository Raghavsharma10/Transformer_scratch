def show_download_links(self):
        """
        Query PyPI for pkg download URI for a packge

        @returns: 0

        """
        #In case they specify version as 'dev' instead of using -T svn,
        #don't show three svn URI's
        if self.options.file_type == "all" and self.version == "dev":
            self.options.file_type = "svn"

        if self.options.file_type == "svn":
            version = "dev"
        else:
            if self.version:
                version = self.version
            else:
                version = self.all_versions[0]
        if self.options.file_type == "all":
            #Search for source, egg, and svn
            self.print_download_uri(version, True)
            self.print_download_uri(version, False)
            self.print_download_uri("dev", True)
        else:
            if self.options.file_type == "source":
                source = True
            else:
                source = False
            self.print_download_uri(version, source)
        return 0