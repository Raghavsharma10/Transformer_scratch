def swagger_schema(self, request):
        """Render API Schema."""
        if self.parent is None:
            return {}

        spec = APISpec(
            self.parent.name, self.parent.cfg.get('VERSION', ''),
            plugins=['apispec.ext.marshmallow'], basePatch=self.prefix
        )

        for paths, handler in self.handlers.items():
            spec.add_tag({
                'name': handler.name,
                'description': utils.dedent(handler.__doc__ or ''),
            })
            for path in paths:
                operations = {}
                for http_method in handler.methods:
                    method = getattr(handler, http_method.lower())
                    operation = OrderedDict({
                        'tags': [handler.name],
                        'summary': method.__doc__,
                        'produces': ['application/json'],
                        'responses': {200: {'schema': {'$ref': {'#/definitions/' + handler.name}}}}
                    })
                    operation.update(utils.load_yaml_from_docstring(method.__doc__) or {})
                    operations[http_method.lower()] = operation

                spec.add_path(self.prefix + path, operations=operations)

            if getattr(handler, 'Schema', None):
                kwargs = {}
                if getattr(handler.meta, 'model', None):
                    kwargs['description'] = utils.dedent(handler.meta.model.__doc__ or '')
                spec.definition(handler.name, schema=handler.Schema, **kwargs)

        return deepcopy(spec.to_dict())