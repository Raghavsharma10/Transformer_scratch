def load_uri(self, uri, verbose):
        """
        
        :param uri:
        :param rdf_format_opts:
        :param verbose:
        :return:
        """

        if verbose: printDebug("----------")
        if verbose: printDebug("Reading: <%s>" % uri)
        success = False
        for f in self.rdf_format_opts:
            if verbose: printDebug(".. trying rdf serialization: <%s>" % f)
            try:
                self.rdfgraph.parse(uri, format=f)
                if verbose: printDebug("..... success!", bold=True)
                success = True
                self.sources_valid += [uri]
                break
            except:
                if verbose: printDebug("..... failed")

        if not success == True:
            self.loading_failed(self.rdf_format_opts)
            self.sources_invalid += [uri]