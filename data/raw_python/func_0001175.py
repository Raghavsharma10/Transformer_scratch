def create_ssl_certs(self):
        """
        Creates SSL cert files
        """
        for key, file in self.ssl.items():
            file["file"] = self.create_temp_file(file["suffix"], file["content"])