def get_DOI(self):
        """
        This method defines how the Article tries to detect the DOI.

        It attempts to determine the article DOI string by DTD-appropriate
        inspection of the article metadata. This method should be made as
        flexible as necessary to properly collect the DOI for any XML
        publishing specification.

        Returns
        -------
        doi : str or None
            The full (publisher/article) DOI string for the article, or None on
            failure.
        """
        if self.dtd_name == 'JPTS':
            doi = self.root.xpath("./front/article-meta/article-id[@pub-id-type='doi']")
            if doi:
                return doi[0].text
            log.warning('Unable to locate DOI string for this article')
            return None
        else:
            log.warning('Unable to locate DOI string for this article')
            return None