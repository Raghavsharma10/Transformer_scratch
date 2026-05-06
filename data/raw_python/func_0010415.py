def _read_descriptions(self, password):
        """
        Read and evaluate the igddesc.xml file
        and the tr64desc.xml file if a password is given.
        """
        descfiles = [FRITZ_IGD_DESC_FILE]
        if password:
            descfiles.append(FRITZ_TR64_DESC_FILE)
        for descfile in descfiles:
            parser = FritzDescParser(self.address, self.port, descfile)
            if not self.modelname:
                self.modelname = parser.get_modelname()
            services = parser.get_services()
            self._read_services(services)