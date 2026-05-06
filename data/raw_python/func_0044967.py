def create_schema(self, model, waiting_models):
        """
        Creates search schemas.

        Args:
            model: model to execute
            waiting_models: if riak can't return response immediately, model is taken to queue.
            After first execution session, method is executed with waiting models and controlled.
            And be ensured that all given models are executed properly.

        Returns:

        """
        bucket_name = model._get_bucket_name()
        index_name = "%s_%s" % (settings.DEFAULT_BUCKET_TYPE, bucket_name)
        ins = model(fake_context)
        fields = self.get_schema_fields(ins._collect_index_fields())
        new_schema = self.compile_schema(fields)
        schema = get_schema_from_solr(index_name)
        if not (schema == new_schema):
            try:
                client.create_search_schema(index_name, new_schema)
                print("+ %s (%s) search schema is created." % (model.__name__, index_name))
            except:
                print("+ %s (%s) search schema checking operation is taken to queue." % (
                    model.__name__, index_name))
                waiting_models.append(model)