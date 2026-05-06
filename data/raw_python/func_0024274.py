def check_migration_and_solr(self):
        """
            The model or models are checked for migrations that need to be done.
            Solr is also checked.
        """
        from pyoko.db.schema_update import SchemaUpdater
        from socket import error as socket_error
        from pyoko.conf import settings
        from importlib import import_module

        import_module(settings.MODELS_MODULE)
        registry = import_module('pyoko.model').model_registry
        models = [model for model in registry.get_base_models()]
        try:
            print(__(u"Checking migration and solr ..."))
            updater = SchemaUpdater(models, 1, False)
            updater.run(check_only=True)

        except socket_error as e:
            print(__(u"{0}Error not connected, open redis and rabbitmq{1}").format(CheckList.FAIL,
                                                                                   CheckList.ENDC))