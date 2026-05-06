def init_soap_exception(self, exc):
        """
        Initializes exception based on soap error response.
        @param exc: URLError
        @return: SoapException
        """
        if not isinstance(exc, urllib2.HTTPError):
            return SoapException(unicode(exc), exc)

        if isinstance(exc, urllib2.HTTPError):
            try:
                data = exc.read()
                self.log.debug(data)

                t = SOAPpy.Parser.parseSOAP(data)
                message = '%s:%s' % (t.Fault.faultcode, t.Fault.faultstring)
                e = SoapException(message, exc)
                e.code = t.Fault.detail.Error.Code
                e.trace = t.Fault.detail.Error.Trace
                return e
            except:
                return SoapException(unicode(exc), exc)

        return SoapException(exc.reason, exc)