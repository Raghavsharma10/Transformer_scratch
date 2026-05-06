def formfield_for_dbfield(self, db_field, **kwargs):
        """
        Hook for specifying the form Field instance for a given
        database Field instance. If kwargs are given, they're
        passed to the form Field's constructor.

        Default implementation uses the overrides returned by
        `get_formfield_overrides`. If a widget is an instance
        of APIChoiceWidget this will do lookup on the current
        admin site for the bundle that is registered for that
        module as the primary bundle for that one model. If a
        match is found then this will call update_links on that
        widget to store the appropriate urls for the javascript
        to call. Otherwise the widget is removed and the default
        select widget will be used instead.
        """

        overides = self.get_formfield_overrides()

        # If we've got overrides for the formfield defined, use 'em. **kwargs
        # passed to formfield_for_dbfield override the defaults.
        for klass in db_field.__class__.mro():
            if klass in overides:
                kwargs = dict(overides[klass], **kwargs)
                break

        # Our custom widgets need special init
        mbundle = None
        extra = kwargs.pop('widget_kwargs', {})
        widget = kwargs.get('widget')
        if kwargs.get('widget'):
            if widget and isinstance(widget, type) and \
                            issubclass(widget, widgets.APIChoiceWidget):
                mbundle = self.bundle.admin_site.get_bundle_for_model(
                                                db_field.rel.to)
                if mbundle:
                    widget = widget(db_field.rel, **extra)
                else:
                    widget = None

        if getattr(self, 'prepopulated_fields', None) and \
                        not getattr(self, 'object', None) and \
                        db_field.name in self.prepopulated_fields:
            extra = kwargs.pop('widget_kwargs', {})
            attr = extra.pop('attrs', {})
            attr['data-source-fields'] = self.prepopulated_fields[db_field.name]
            extra['attrs'] = attr
            if not widget:
                from django.forms.widgets import TextInput
                widget = TextInput(**extra)
            elif widget and isinstance(widget, type):
                widget = widget(**extra)

        kwargs['widget'] = widget

        field = db_field.formfield(**kwargs)
        if mbundle:
            field.widget.update_links(self.request, self.bundle.admin_site)
        return field