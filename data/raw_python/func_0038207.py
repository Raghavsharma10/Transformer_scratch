def list(self, request):
        """Search the doctypes for this model."""
        query = get_query_params(request).get("search", "")
        results = []
        base = self.model.get_base_class()
        doctypes = indexable_registry.families[base]
        for doctype, klass in doctypes.items():
            name = klass._meta.verbose_name.title()
            if query.lower() in name.lower():
                results.append(dict(
                    name=name,
                    doctype=doctype
                ))
                results.sort(key=lambda x: x["name"])
        return Response(dict(results=results))