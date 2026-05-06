def render_to_response(self, context=None, request=None, def_name=None, content_type=None, status=None, charset=None):
        '''
        Renders the template and returns an HttpRequest object containing its content.

        This method returns a django.http.Http404 exception if the template is not found.
        If the template raises a django_mako_plus.RedirectException, the browser is redirected to
        the given page, and a new request from the browser restarts the entire DMP routing process.
        If the template raises a django_mako_plus.InternalRedirectException, the entire DMP
        routing process is restarted internally (the browser doesn't see the redirect).

            @request      The request context from Django.  If this is None, any TEMPLATE_CONTEXT_PROCESSORS defined in your settings
                          file will be ignored but the template will otherwise render fine.
            @template     The template file path to render.  This is relative to the app_path/controller_TEMPLATES_DIR/ directory.
                          For example, to render app_path/templates/page1, set template="page1.html", assuming you have
                          set up the variables as described in the documentation above.
            @context      A dictionary of name=value variables to send to the template page.  This can be a real dictionary
                          or a Django Context object.
            @def_name     Limits output to a specific top-level Mako <%block> or <%def> section within the template.
                          For example, def_name="foo" will call <%block name="foo"></%block> or <%def name="foo()"></def> within the template.
            @content_type The MIME type of the response.  Defaults to settings.DEFAULT_CONTENT_TYPE (usually 'text/html').
            @status       The HTTP response status code.  Defaults to 200 (OK).
            @charset      The charset to encode the processed template string (the output) with.  Defaults to settings.DEFAULT_CHARSET (usually 'utf-8').

        The method triggers two signals:
            1. dmp_signal_pre_render_template: you can (optionally) return a new Mako Template object from a receiver to replace
               the normal template object that is used for the render operation.
            2. dmp_signal_post_render_template: you can (optionally) return a string to replace the string from the normal
               template object render.
        '''
        try:
            if content_type is None:
                content_type = mimetypes.types_map.get(os.path.splitext(self.mako_template.filename)[1].lower(), settings.DEFAULT_CONTENT_TYPE)
            if charset is None:
                charset = settings.DEFAULT_CHARSET
            if status is None:
                status = 200
            content = self.render(context=context, request=request, def_name=def_name)
            return HttpResponse(content.encode(charset), content_type='%s; charset=%s' % (content_type, charset), status=status)

        except RedirectException: # redirect to another page
            e = sys.exc_info()[1]
            if request is None:
                log.info('a template redirected processing to %s', e.redirect_to)
            else:
                log.info('view function %s.%s redirected processing to %s', request.dmp.module, request.dmp.function, e.redirect_to)
            # send the signal
            dmp = apps.get_app_config('django_mako_plus')
            if dmp.options['SIGNALS']:
                dmp_signal_redirect_exception.send(sender=sys.modules[__name__], request=request, exc=e)
            # send the browser the redirect command
            return e.get_response(request)