def as_widget(self, widget=None, attrs=None, only_initial=False):
        """
        Renders the field.
        """
        attrs = attrs or {}
        attrs.update(self.form.get_widget_attrs(self))
        if hasattr(self.field, 'widget_css_classes'):
            css_classes = self.field.widget_css_classes
        else:
            css_classes = getattr(self.form, 'widget_css_classes', None)
        if css_classes:
            attrs.update({'class': css_classes})
        widget_classes = self.form.fields[self.name].widget.attrs.get('class', None)
        if widget_classes:
            if attrs.get('class', None):
                attrs['class'] += ' ' + widget_classes
            else:
                attrs.update({'class': widget_classes})
        return super(NgBoundField, self).as_widget(widget, attrs, only_initial)