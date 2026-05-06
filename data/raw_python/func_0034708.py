def error_handler(self, e, request, meth, em_format):
        """
        Override this method to add handling of errors customized for your 
        needs
        """
        if isinstance(e, FormValidationError):
            return self.form_validation_response(e)

        elif isinstance(e, TypeError):
            result = rc.BAD_REQUEST
            hm = HandlerMethod(meth)
            sig = hm.signature

            msg = 'Method signature does not match.\n\n'

            if sig:
                msg += 'Signature should be: %s' % sig
            else:
                msg += 'Resource does not expect any parameters.'

            if self.display_errors:
                msg += '\n\nException was: %s' % str(e)

            result.content = format_error(msg)
            return result
        elif isinstance(e, Http404):
            return rc.NOT_FOUND

        elif isinstance(e, HttpStatusCode):
            return e.response
 
        else: 
            """
            On errors (like code errors), we'd like to be able to
            give crash reports to both admins and also the calling
            user. There's two setting parameters for this:

            Parameters::
             - `PISTON_EMAIL_ERRORS`: Will send a Django formatted
               error email to people in `settings.ADMINS`.
             - `PISTON_DISPLAY_ERRORS`: Will return a simple traceback
               to the caller, so he can tell you what error they got.

            If `PISTON_DISPLAY_ERRORS` is not enabled, the caller will
            receive a basic "500 Internal Server Error" message.
            """
            exc_type, exc_value, tb = sys.exc_info()
            rep = ExceptionReporter(request, exc_type, exc_value, tb.tb_next)
            if self.email_errors:
                self.email_exception(rep)
            if self.display_errors:
                return HttpResponseServerError(
                    format_error('\n'.join(rep.format_exception())))
            else:
                raise