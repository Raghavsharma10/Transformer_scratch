def process_exception(self, request, e):
        """
        Logs exception error message and sends email to ADMINS if hostname is not testserver and DEBUG=False.
        :param request: HttpRequest
        :param e: Exception
        """
        from jutil.email import send_email

        assert isinstance(request, HttpRequest)
        full_path = request.get_full_path()
        user = request.user
        msg = '{full_path}\n{err} (IP={ip}, user={user}) {trace}'.format(full_path=full_path, user=user, ip=get_real_ip(request), err=e, trace=str(traceback.format_exc()))
        logger.error(msg)
        hostname = request.get_host()
        if not settings.DEBUG and hostname != 'testserver':
            send_email(settings.ADMINS, 'Error @ {}'.format(hostname), msg)
        return None