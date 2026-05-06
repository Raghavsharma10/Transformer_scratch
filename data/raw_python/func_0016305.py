def sslErrors(self, ssl_errors):
        """
        Handle SSL errors, logging them if debug is on and ignoring them
        if disable_ssl_certificate_validation is set.
        """
        if ssl_errors:
            for v in ssl_errors:
                self.msg_log("SSL Error: %s" % v.errorString())
        if self.disable_ssl_certificate_validation:
            self.reply.ignoreSslErrors()