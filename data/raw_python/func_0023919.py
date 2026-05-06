def get_context_data(self, **kwargs):
        '''
        Generic list view with validation included and object transfering support
        '''

        # Call the base implementation first to get a context
        context = super(GenList, self).get_context_data(**kwargs)
        # Update general context with the stuff we already calculated
        context.update(self.__context)

        # Initialize with our timestamp
        context['now'] = epochdate(time.time())
        context['profile'] = self.profile

        # Check vtable
        context['vtable'] = getattr(self, 'vtable', False)

        # Export to excel
        context['export_excel'] = getattr(self, 'export_excel', True)
        context['export_name'] = getattr(self, 'export_name', 'list')

        # Check ngincludes
        context['ngincludes'] = getattr(self, 'ngincludes', {})
        if 'table' not in context['ngincludes'].keys():
            context['ngincludes']['table'] = "{}codenerix/partials/table.html".format(settings.STATIC_URL)

        # Check linkadd
        context['linkadd'] = getattr(self, 'linkadd', self.auth_permission('add') or getattr(self, 'public', False))

        # Check linkedit
        context['linkedit'] = getattr(self, 'linkedit', self.auth_permission('change') or getattr(self, 'public', False))

        # Check showdetails
        context['show_details'] = getattr(self, 'show_details', False)

        # Check showmodal
        context['show_modal'] = getattr(self, 'show_modal', False)

        # Set search filter button
        context['search_filter_button'] = getattr(self, 'search_filter_button', False)

        # Get base template
        if not self.json_worker:
            template_base = getattr(self, 'template_base', 'base/base')
            template_base_ext = getattr(self, 'template_base_ext', 'html')
            context['template_base'] = get_template(template_base, self.user, self.language, extension=template_base_ext)

        # Try to convert object_id to a numeric id
        object_id = kwargs.get('object_id', None)
        try:
            object_id = int(object_id)
        except Exception:
            pass

        # Python 2 VS Python 3 compatibility
        try:
            unicode('codenerix')
            unicodetest = unicode
        except NameError:
            unicodetest = str

        if isinstance(object_id, str) or isinstance(object_id, unicodetest):
            # If object_id is a string, we have a name not an object
            context['object_name'] = object_id
            object_obj = None
        else:
            # If is not an string
            if object_id:
                # If we got one, load the object
                obj = context['obj']
                object_obj = get_object_or_404(obj, pk=object_id)
            else:
                # There is no object
                object_obj = None
            context['object_obj'] = object_obj

        # Attach extra_context
        context.update(self.extra_context)
        # Return new context
        return context