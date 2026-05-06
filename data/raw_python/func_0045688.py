def load_file(self, file_obj, verbose):
        """
        The type of open file objects such as sys.stdout; alias of the built-in file.
        @TODO: when is this used? 
        """
        if verbose: printDebug("----------")
        if verbose: printDebug("Reading: <%s> ...'" % file_obj.name)

        if type(file_obj) == file:
            self.rdfgraph = self.rdfgraph + file_obj
            self.sources_valid += [file_obj.NAME]
        else:
            self.loading_failed(self.rdf_format_opts)
            self.sources_invalid += [file_obj.NAME]