def get_parent_object(self):
        """
        Lookup a parent object. If parent_field is None
        this will return None. Otherwise this will try to
        return that object.

        The filter arguments are found by using the known url
        parameters of the bundle, finding the value in the url keyword
        arguments and matching them with the arguments in
        `self.parent_lookups`. The first argument in parent_lookups
        matched with the value of the last argument in the list of bundle
        url parameters, the second with the second last and so forth.

        For example let's say the parent_field attribute is 'gallery'
        and the current bundle knows about these url parameters:

        * adm_post
        * adm_post_gallery

        And the current value for 'self.kwargs' is:

        * adm_post = 2
        * adm_post_gallery = 3

        if parent_lookups isn't set the filter for the queryset
        on the gallery model will be:

        * pk = 3

        if parent_lookups is ('pk', 'post__pk') then the filter
        on the queryset will be:

        * pk = 3
        * post__pk = 2

        The model to filter on is found by finding the relationship
        in self.parent_field and filtering on that model.
        If a match is found, 'self.queryset` is changed to
        filter on the parent as described above and the parent
        object is returned. If no match is found, a Http404 error
        is raised.
        """

        if self.parent_field:
            # Get the model we are querying on
            if getattr(self.model._meta, 'init_name_map', None):
                # pre-django-1.8
                cache = self.model._meta.init_name_map()
                field, mod, direct, m2m = cache[self.parent_field]
            else:
                # 1.10
                if DJANGO_VERSION[1] >= 10:
                    field = self.model._meta.get_field(self.parent_field)
                    m2m = field.is_relation and field.many_to_many
                    direct = not field.auto_created or field.concrete
                else:
                    # 1.8 and 1.9
                    field, mod, direct, m2m = self.model._meta.get_field(self.parent_field)

            to = None
            field_name = None
            if self.parent_lookups is None:
                self.parent_lookups = ('pk',)

            url_params = list(self.bundle.url_params)
            if url_params and getattr(self.bundle, 'delegated', False):
                url_params = url_params[:-1]

            offset = len(url_params) - len(self.parent_lookups)
            kwargs = {}
            for i in range(len(self.parent_lookups) - 1):
                k = url_params[offset + i]
                value = self.kwargs[k]
                kwargs[self.parent_lookups[i + 1]] = value

            main_arg = self.kwargs[url_params[-1]]
            main_key = self.parent_lookups[0]

            if m2m:
                rel = getattr(self.model, self.parent_field)
                kwargs[main_key] = main_arg
                if direct:
                    to = rel.field.rel.to
                    field_name = self.parent_field
                else:
                    try:
                        from django.db.models.fields.related import (
                            ForeignObjectRel)
                        if isinstance(rel.rel, ForeignObjectRel):
                            to = rel.rel.related_model
                        else:
                            to = rel.rel.model
                    except ImportError:
                        to = rel.rel.model
                    field_name = rel.rel.field.name
            else:
                to = field.rel.to
                if main_key == 'pk':
                    to_field = field.rel.field_name
                    if to_field == 'vid':
                        to_field = 'object_id'
                else:
                    to_field = main_key
                kwargs[to_field] = main_arg

            # Build the list of arguments
            try:
                obj = to.objects.get(**kwargs)
                if self.queryset is None:
                    if m2m:
                        self.queryset = getattr(obj, field_name)
                    else:
                        self.queryset = self.model.objects.filter(
                                                    **{self.parent_field: obj})
                return obj
            except to.DoesNotExist:
                raise http.Http404
        return None