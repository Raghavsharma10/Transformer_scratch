def _get_instances(cls, path, context=None, site=None, language=None):
        """ A sequence of instances to discover metadata. 
            Each instance from each backend is looked up when possible/necessary.
            This is a generator to eliminate unnecessary queries.
        """
        backend_context = {'view_context': context }

        for model in cls._meta.models.values():
            for instance in model.objects.get_instances(path, site, language, backend_context) or []:
                if hasattr(instance, '_process_context'):
                    instance._process_context(backend_context)
                yield instance