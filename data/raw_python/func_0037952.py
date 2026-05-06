def body_template(self, value):
        """
        Must be an instance of a prestans.types.DataCollection subclass; this is
        generally set during the RequestHandler lifecycle. Setting this spwans the
        parsing process of the body. If the HTTP verb is GET an AssertionError is
        thrown. Use with extreme caution.
        """

        if self.method == VERB.GET:
            raise AssertionError("body_template cannot be set for GET requests")

        if value is None:
            self.logger.warning("body_template is None, parsing will be ignored")
            return

        if not isinstance(value, DataCollection):
            msg = "body_template must be an instance of %s.%s" % (
                DataCollection.__module__,
                DataCollection.__name__
            )
            raise AssertionError(msg)

        self._body_template = value

        # get a deserializer based on the Content-Type header
        # do this here so the handler gets a chance to setup extra serializers
        self.set_deserializer_by_mime_type(self.content_type)