def inspect_model(self, model):
        """
        Inspect a single model
        """
        # See which interesting fields the model holds.
        url_fields = sorted(f for f in model._meta.fields if isinstance(f, (PluginUrlField, models.URLField)))
        file_fields = sorted(f for f in model._meta.fields if isinstance(f, (PluginImageField, models.FileField)))
        html_fields = sorted(f for f in model._meta.fields if isinstance(f, (models.TextField, PluginHtmlField)))
        all_fields = [f.name for f in (file_fields + html_fields + url_fields)]
        if not all_fields:
            return []

        if model.__name__ in self.exclude:
            self.stderr.write("Skipping {0} ({1})\n".format(model.__name__, ", ".join(all_fields)))
            return []

        sys.stderr.write("Inspecting {0} ({1})\n".format(model.__name__, ", ".join(all_fields)))

        q_notnull = reduce(operator.or_, (Q(**{"{0}__isnull".format(f): False}) for f in all_fields))
        qs = model.objects.filter(q_notnull).order_by('pk')

        urls = []
        for object in qs:
            # HTML fields need proper html5lib parsing
            for field in html_fields:
                value = getattr(object, field.name)
                if value:
                    html_images = self.extract_html_urls(value)
                    urls += html_images

                    for image in html_images:
                        self.show_match(object, image)

            # Picture fields take the URL from the storage class.
            for field in file_fields:
                value = getattr(object, field.name)
                if value:
                    value = unquote_utf8(value.url)
                    urls.append(value)
                    self.show_match(object, value)

            # URL fields can be read directly.
            for field in url_fields:
                value = getattr(object, field.name)
                if value:
                    if isinstance(value, six.text_type):
                        value = force_text(value)
                    else:
                        value = value.to_db_value()  # AnyUrlValue

                    urls.append(value)
                    self.show_match(object, value)
        return urls