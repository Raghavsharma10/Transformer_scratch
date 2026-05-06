def setup_mappings(cls, force=False):
        """ Setup ES mappings for all existing models.

        This method is meant to be run once at application lauch.
        ES._mappings_setup flag is set to not run make mapping creation
        calls on subsequent runs.

        Use `force=True` to make subsequent calls perform mapping
        creation calls to ES.
        """
        if getattr(cls, '_mappings_setup', False) and not force:
            log.debug('ES mappings have been already set up for currently '
                      'running application. Call `setup_mappings` with '
                      '`force=True` to perform mappings set up again.')
            return
        log.info('Setting up ES mappings for all existing models')
        models = engine.get_document_classes()
        try:
            for model_name, model_cls in models.items():
                if getattr(model_cls, '_index_enabled', False):
                    es = cls(model_cls.__name__)
                    es.put_mapping(body=model_cls.get_es_mapping())
        except JHTTPBadRequest as ex:
            raise Exception(ex.json['extra']['data'])
        cls._mappings_setup = True