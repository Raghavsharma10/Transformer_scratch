def get(self, request, *args, **kwargs):
        """
        Handles GET requests and instantiates blank versions of the form and its inline formsets.
        """

        # Prepare base
        if 'pk' in kwargs:
            self.object = self.get_object()
        else:
            self.object = None
        form_class = self.get_form_class()

        # Get prefix
        if 'field_prefix' in form_class.Meta.__dict__:
            # Get name from the form
            field_prefix = form_class.Meta.field_prefix
        else:
            # Get name from the class
            field_prefix = str(form_class).split("'")[1].split(".")[-1]
        self.field_prefix = field_prefix

        # Build form
        form = self.get_form(form_class)

        # Find groups
        if 'groups' in dir(self):
            # Save groups
            groups = self.groups
            # Redefine groups inside the form
            form.__groups__ = lambda: groups
            # Initialize list of fields
            fields = []
        else:
            groups = None

        # Add special prefix support to properly support form independency
        form.add_prefix = lambda fields_name, field_prefix=field_prefix: "%s_%s" % (field_prefix, fields_name)
        if 'autofill' not in dir(form.Meta):
            form.Meta.autofill = {}

        # For every extra form
        forms = []
        position_form_default = 0
        for (formelement, linkerfield, modelfilter) in self.forms:
            if formelement is None:
                formobj = form
                position_form_default = len(forms)
            else:
                # Locate linked element
                if self.object:
                    related_name = formelement._meta.model._meta.get_field(linkerfield).related_query_name()
                    queryset = getattr(self.object, related_name)
                    if modelfilter:
                        queryset = queryset.filter(eval("Q(%s)" % (modelfilter)))
                    get_method = getattr(queryset, 'get', None)
                    if get_method:
                        instance = queryset.get()
                    else:
                        instance = queryset
                else:
                    instance = None

                if 'autofill' in dir(formelement.Meta):
                    formname = str(formelement).split('.')[-1].split("'")[0]
                    for key in formelement.Meta.autofill:
                        form.Meta.autofill['{}_{}'.format(formname, key)] = formelement.Meta.autofill[key]

                # Get prefix
                if 'field_prefix' in formelement.Meta.__dict__:
                    # Get name from the form
                    field_prefix = formelement.Meta.field_prefix
                else:
                    # Get name from the class
                    field_prefix = str(formelement).split("'")[1].split(".")[-1]
                self.field_prefix = field_prefix

                # Prepare form
                formobj = formelement(instance=instance)
                formobj.form_name = form.form_name

                # Excluded fields
                if 'exclude' not in formobj.Meta.__dict__:
                    formobj.Meta.exclude = [linkerfield]
                elif linkerfield not in formobj.Meta.exclude:
                    formobj.Meta.exclude.append(linkerfield)
                if linkerfield in formobj.fields:
                    del(formobj.fields[linkerfield])

                # Add special prefix support to properly support form independency
                formobj.add_prefix = lambda fields_name, field_prefix=field_prefix: "%s_%s" % (field_prefix, fields_name)
                formobj.scope_prefix = field_prefix

            # Save fields to the list
            if groups:
                for field in formobj:
                    fields.append(field)
            else:
                # Add the form to the list of forms
                forms.append(formobj)

        if position_form_default == 0:
            open_tabs = 1
        else:
            open_tabs = 0
        # Remember list of fields
        if groups:
            form.list_fields = fields

        # Add context and return new context
        return self.render_to_response(self.get_context_data(form=form, forms=forms, open_tabs=open_tabs, position_form_default=position_form_default))