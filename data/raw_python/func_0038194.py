def get_serializer_class(self):
        """gets the class type of the serializer

        :return: `rest_framework.Serializer`
        """
        klass = None

        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field
        if lookup_url_kwarg in self.kwargs:
            # Looks like this is a detail...
            klass = self.get_object().__class__
        elif "doctype" in self.request.REQUEST:
            base = self.model.get_base_class()
            doctypes = indexable_registry.families[base]
            try:
                klass = doctypes[self.request.REQUEST["doctype"]]
            except KeyError:
                raise Http404

        if hasattr(klass, "get_serializer_class"):
            return klass.get_serializer_class()

        # TODO: fix deprecation warning here -- `get_serializer_class` is going away soon!
        return super(ContentViewSet, self).get_serializer_class()