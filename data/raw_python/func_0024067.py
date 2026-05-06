def post(self, request, *args, **kwargs):
        """
        andles POST requests, instantiating a form instance and its inline formsets with the passed POST variables and then checking them for validity.
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
            # Initialize list of fields
        else:
            # Get name from the class
            field_prefix = str(form_class).split("'")[1].split(".")[-1]
        self.field_prefix = field_prefix

        # Build the form
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

        # Check validation
        valid = form.is_valid()
        if (not valid) and ('non_field_errors' in dir(self)):
            errors = [element[5] for element in list(self.non_field_errors())[:-1]]
        elif form.errors.as_data():
            errors = []
            for element in form.errors.as_data():
                for err in form.errors.as_data()[element][0]:
                    errors.append(err)
        else:
            errors = []

        # For every extra form
        temp_forms = []
        position_form_default = 0
        for (formelement, linkerfield, modelfilter) in self.forms:
            if formelement is None:
                formobj = form
                position_form_default = len(temp_forms)
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

                # Get prefix
                if 'field_prefix' in formelement.Meta.__dict__:
                    # Get name from the form
                    field_prefix = formelement.Meta.field_prefix
                else:
                    # Get name from the class
                    field_prefix = str(formelement).split("'")[1].split(".")[-1]
                self.field_prefix = field_prefix

                # Prepare form
                formobj = formelement(instance=instance, data=self.request.POST)
                formobj.form_name = form.form_name

                # Excluded fields
                if 'exclude' not in formobj.Meta.__dict__:
                    formobj.Meta.exclude = [linkerfield]
                elif linkerfield not in formobj.Meta.exclude:
                    formobj.Meta.exclude.append(linkerfield)
                if linkerfield in formobj.fields:
                    del(formobj.fields[linkerfield])

                # Link it to the main model
                formobj.add_prefix = lambda fields_name, field_prefix=field_prefix: "%s_%s" % (field_prefix, fields_name)

                # Validate
                valid *= formobj.is_valid()

                # append error
                if not formobj.is_valid() and ('non_field_errors' in dir(formobj)):
                    errors += [element[5] for element in list(formobj.non_field_errors())[:-1]]

            # Save fields to the list
            if groups:
                for field in formobj:
                    # raise Exception (field.__dict__)
                    if 'unblock_t2ime' in field.html_name:
                        raise Exception(field.field.__dict__)
                    fields.append(field)

            # Add a new form
            temp_forms.append((formobj, linkerfield))

        # execute validation specified
        validate_forms = None
        if valid and ("validate" in dir(self)):
            validate_forms = [tform[0] for tform in temp_forms]
            errors = self.validate(*validate_forms)
            # valid = len(errors) == 0
            valid = False
            if errors is None or len(errors) == 0:
                valid = True

        # Remember list of fields
        if groups:
            form.list_fields = fields
            forms = []
        else:
            if validate_forms:
                forms = validate_forms
            else:
                forms = [tform[0] for tform in temp_forms]

        if position_form_default == 0:
            open_tabs = 1
        else:
            open_tabs = 0

        # Check validation result
        if valid:
            # Everything is OK, call valid
            return self.form_valid(form, temp_forms)
        else:
            # Something went wrong, attach error and call invalid
            form.list_errors = errors
            return self.form_invalid(form, forms, open_tabs, position_form_default)