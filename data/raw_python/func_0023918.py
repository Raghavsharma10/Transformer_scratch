def dispatch(self, *args, **kwargs):
        '''
        Entry point for this class, here we decide basic stuff
        '''

        # Get if this class is working as only a base render and List funcionality shouldn't be enabled
        onlybase = getattr(self, "onlybase", False)

        # REST not available when onlybase is enabled
        if not onlybase:

            # Check if this is a REST query to pusth the answer to responde in JSON
            if bool(self.request.META.get('HTTP_X_REST', False)):
                self.json = True
                if self.request.GET.get('json', self.request.POST.get('json', None)) is None:
                    newget = {}
                    newget['json'] = "{}"
                    for key in self.request.GET:
                        newget[key] = self.request.GET[key]
                    self.request.GET = QueryDict('').copy()
                    self.request.GET.update(newget)

    #                return HttpResponseBadRequest(_("The service requires you to set a GET argument named json={} which will contains all the filters you can apply to a list"))

            # Check if this is a REST query to add an element
            if self.request.method == 'POST':
                target = get_class(resolve("{}/add".format(self.request.META.get("REQUEST_URI"))).func)
                target.json = True
                return target.as_view()(self.request)

        # Set class internal variables
        self._setup(self.request)

        # Deprecations
        deprecated = [('retrictions', '2016061000')]
        for (depre, version) in deprecated:
            if hasattr(self, depre):
                raise IOError("The attribute '{}' has been deprecated in version '{}' and it is not available anymore".format(version))

        # Build extracontext
        if not hasattr(self, 'extra_context'):
            self.extra_context = {}
        if not hasattr(self, 'client_context'):
            self.client_context = {}
        # Attach user to the extra_context
        self.extra_context['user'] = self.user

        # Attach WS entry point and STATIC entry point
        self.extra_context['ws_entry_point'] = self.BASE_URL + getattr(self, "ws_entry_point", "{0}/{1}".format(self._appname, "{0}s".format(self._modelname.lower())))

        static_partial_row_path = getattr(self, 'static_partial_row', "{0}/{1}_rows.html".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_partial_row'] = get_static(static_partial_row_path, self.user, self.language, self.DEFAULT_STATIC_PARTIAL_ROWS, 'html', relative=True)

        static_partial_header_path = getattr(self, 'static_partial_header', "{0}/{1}_header.html".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_partial_header'] = get_static(static_partial_header_path, self.user, self.language, None, 'html', relative=True)

        static_partial_summary_path = getattr(self, 'static_partial_summary', "{0}/{1}_summary.html".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_partial_summary'] = get_static(static_partial_summary_path, self.user, self.language, self.DEFAULT_STATIC_PARTIAL_SUMMARY, 'html', relative=True)

        static_app_row_path = getattr(self, 'static_app_row', "{0}/{1}_app.js".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_app_row'] = get_static(static_app_row_path, self.user, self.language, os.path.join(settings.STATIC_URL, 'codenerix/js/app.js'), 'js', relative=True)

        static_controllers_row_path = getattr(self, 'static_controllers_row', "{0}/{1}_controllers.js".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_controllers_row'] = get_static(static_controllers_row_path, self.user, self.language, None, 'js', relative=True)

        static_filters_row_path = getattr(self, 'static_filters_row', "{0}/{1}_filters.js".format(self._appname, "{0}s".format(self._modelname.lower())))
        self.extra_context['static_filters_row'] = get_static(static_filters_row_path, self.user, self.language, os.path.join(settings.STATIC_URL, 'codenerix/js/rows.js'), 'js', relative=True)

        self.extra_context['field_delete'] = getattr(self, 'field_delete', False)
        self.extra_context['field_check'] = getattr(self, 'field_check', None)

        # Default value for extends_base
        if hasattr(self, 'extends_base'):
            self.extra_context['extends_base'] = self.extends_base
        elif hasattr(self, 'extends_base'):
            self.extra_context['extends_base'] = self.extends_base

        # Get if this is a template only answer
        self.__authtoken = (bool(getattr(self.request, "authtoken", False)))
        self.json_worker = (hasattr(self, 'json_builder')) or self.__authtoken or (self.json is True)
        if self.json_worker:
            # Check if the request has some json query, if not, just render the template
            if self.request.GET.get('json', self.request.POST.get('json', None)) is None:
                # Calculate tabs
                if getattr(self, 'show_details', False):
                    self.extra_context['tabs_js'] = json.dumps(self.get_tabs_js())

                # Silence the normal execution from this class
                self.get_queryset = lambda: None
                self.get_context_data = lambda **kwargs: self.extra_context
                self.render_to_response = lambda context, **response_kwargs: super(GenList, self).render_to_response(context, **response_kwargs)
                # Call the base implementation and finish execution here
                return super(GenList, self).dispatch(*args, **kwargs)

        # The systems is requesting a list, we are not allowed
        if onlybase:
            json_answer = {"error": True, "errortxt": _("Not allowed, this kind of requests has been prohibited for this view!")}
            return HttpResponse(json.dumps(json_answer), content_type='application/json')

        # Initialize a default context
        self.__kwargs = kwargs
        self.__context = {}

        # Force export list
        self.export = getattr(self, 'export', self.request.GET.get('export', self.request.POST.get('export', None)))

        # Call the base implementation
        return super(GenList, self).dispatch(*args, **kwargs)