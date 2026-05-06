def get_publisher(self):
        """
        This method defines how the Article tries to determine the publisher of
        the article.

        This method relies on the success of the get_DOI method to fetch the
        appropriate full DOI for the article. It then takes the DOI prefix
        which corresponds to the publisher and then uses that to attempt to load
        the correct publisher-specific code. This may fail; if the DOI is not
        mapped to a code file, if the DOI is mapped but the code file could not
        be located, or if the mapped code file is malformed then this method
        will issue/log an informative error message and return None. This method
        will not try to infer the publisher based on any metadata other than the
        DOI of the article.

        Returns
        -------
        publisher : Publisher instance or None
        """
        #For a detailed explanation of the DOI system, visit:
        #http://www.doi.org/hb.html
        #The basic syntax of a DOI is this <prefix>/<suffix>
        #The <prefix> specifies a unique DOI registrant, in our case, this
        #should correspond to the publisher. We use this information to register
        #the correct Publisher class with this article
        doi_prefix = self.doi.split('/')[0]
        #The import_by_doi method should raise ImportError if a problem occurred
        try:
            publisher_mod = openaccess_epub.publisher.import_by_doi(doi_prefix)
        except ImportError as e:
            log.exception(e)
            return None
        #Each publisher module should define an attribute "pub_class" pointing
        #to the publisher-specific class extending
        #openaccess_epub.publisher.Publisher
        return publisher_mod.pub_class(self)