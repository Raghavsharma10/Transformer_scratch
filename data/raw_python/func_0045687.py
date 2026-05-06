def load_text(self, text, verbose):
        """
        
        :param text:
        :param rdf_format_opts:
        :param verbose:
        :return:
        """
        if verbose: printDebug("----------")
        if verbose: printDebug("Reading: '%s ...'" % text[:10])
        success = False
        for f in self.rdf_format_opts:
            if verbose: printDebug(".. trying rdf serialization: <%s>" % f)
            try:
                self.rdfgraph.parse(data=text, format=f)
                if verbose: printDebug("..... success!")
                success = True
                self.sources_valid += ["Text: '%s ...'" % text[:10]]
                break
            except:
                if verbose: printDebug("..... failed", "error")

        if not success == True:
            self.loading_failed(self.rdf_format_opts)
            self.sources_invalid += ["Text: '%s ...'" % text[:10]]