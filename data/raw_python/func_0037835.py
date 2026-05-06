def _setup_serializers(self):
        """
        Auto set the return serializer based on Accept headers
        http://docs.webob.org/en/latest/reference.html#header-getters

        Intersection of requested types and supported types tells us if we
        can in fact respond in one of the request formats
        """
        acceptable_offers = self.request.accept.acceptable_offers(self.response.supported_mime_types)
        if len(acceptable_offers) > 0:
            best_accept_match = acceptable_offers[0][0]
        else:
            best_accept_match = self.response.default_serializer.content_type()

        # best_accept_match = self.request.accept.best_match(
        #     self.response.supported_mime_types,
        #     default_match=self.response.default_serializer.content_type()
        # )

        self.logger.info("%s determined as best match for accept header: %s" % (
            best_accept_match,
            self.request.accept
        ))

        # if content_type is not acceptable it will raise UnsupportedVocabulary
        self.response.content_type = best_accept_match