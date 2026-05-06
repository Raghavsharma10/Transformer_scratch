def render(self, context=None, request=None, def_name=None):
        '''
        Renders a template using the Mako system.  This method signature conforms to
        the Django template API, which specifies that template.render() returns a string.

            @context  A dictionary of name=value variables to send to the template page.  This can be a real dictionary
                      or a Django Context object.
            @request  The request context from Django.  If this is None, any TEMPLATE_CONTEXT_PROCESSORS defined in your settings
                      file will be ignored but the template will otherwise render fine.
            @def_name Limits output to a specific top-level Mako <%block> or <%def> section within the template.
                      If the section is a <%def>, any parameters must be in the context dictionary.  For example,
                      def_name="foo" will call <%block name="foo"></%block> or <%def name="foo()"></def> within
                      the template.

        Returns the rendered template as a unicode string.

        The method triggers two signals:
            1. dmp_signal_pre_render_template: you can (optionally) return a new Mako Template object from a receiver to replace
               the normal template object that is used for the render operation.
            2. dmp_signal_post_render_template: you can (optionally) return a string to replace the string from the normal
               template object render.
        '''
        dmp = apps.get_app_config('django_mako_plus')
        # set up the context dictionary, which is the variables available throughout the template
        context_dict = {}
        # if request is None, add some default items because the context processors won't happen
        if request is None:
            context_dict['settings'] = settings
            context_dict['STATIC_URL'] = settings.STATIC_URL
        # let the context_processors add variables to the context.
        if not isinstance(context, Context):
            context = Context(context) if request is None else RequestContext(request, context)
        with context.bind_template(self):
            for d in context:
                context_dict.update(d)
        context_dict.pop('self', None)  # some contexts have self in them, and it messes up render_unicode below because we get two selfs

        # send the pre-render signal
        if dmp.options['SIGNALS'] and request is not None:
            for receiver, ret_template_obj in dmp_signal_pre_render_template.send(sender=self, request=request, context=context, template=self.mako_template):
                if ret_template_obj is not None:
                    if isinstance(ret_template_obj, MakoTemplateAdapter):
                        self.mako_template = ret_template_obj.mako_template   # if the signal function sends a MakoTemplateAdapter back, use the real mako template inside of it
                    else:
                        self.mako_template = ret_template_obj                 # if something else, we assume it is a mako.template.Template, so use it as the template

        # do we need to limit down to a specific def?
        # this only finds within the exact template (won't go up the inheritance tree)
        render_obj = self.mako_template
        if def_name is None:
            def_name = self.def_name
        if def_name:  # do we need to limit to just a def?
            render_obj = self.mako_template.get_def(def_name)

        # PRIMARY FUNCTION: render the template
        if log.isEnabledFor(logging.INFO):
            log.info('rendering template %s%s%s', self.name, ('::' if def_name else ''), def_name or '')
        if settings.DEBUG:
            try:
                content = render_obj.render_unicode(**context_dict)
            except Exception as e:
                log.exception('exception raised during template rendering: %s', e)  # to the console
                e.template_debug = get_template_debug('%s%s%s' % (self.name, ('::' if def_name else ''), def_name or ''), e)
                raise
        else:  # this is outside the above "try" loop because in non-DEBUG mode, we want to let the exception throw out of here (without having to re-raise it)
            content = render_obj.render_unicode(**context_dict)

        # send the post-render signal
        if dmp.options['SIGNALS'] and request is not None:
            for receiver, ret_content in dmp_signal_post_render_template.send(sender=self, request=request, context=context, template=self.mako_template, content=content):
                if ret_content is not None:
                    content = ret_content  # sets it to the last non-None return in the signal receiver chain

        # return
        return mark_safe(content)