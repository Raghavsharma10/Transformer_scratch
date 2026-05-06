def create_index(self, model, waiting_models):
        """
        Creates search indexes.

        Args:
            model: model to execute
            waiting_models: if riak can't return response immediately, model is taken to queue.
            After first execution session, method is executed with waiting models and controlled.
            And be ensured that all given models are executed properly.

        Returns:

        """
        bucket_name = model._get_bucket_name()
        bucket_type = client.bucket_type(settings.DEFAULT_BUCKET_TYPE)
        index_name = "%s_%s" % (settings.DEFAULT_BUCKET_TYPE, bucket_name)
        bucket = bucket_type.bucket(bucket_name)
        try:
            client.get_search_index(index_name)
            if not (bucket.get_property('search_index') == index_name):
                bucket.set_property('search_index', index_name)
                print("+ %s (%s) search index is created." % (model.__name__, index_name))
        except RiakError:
            try:
                client.create_search_index(index_name, index_name, self.n_val)
                bucket.set_property('search_index', index_name)
                print("+ %s (%s) search index is created." % (model.__name__, index_name))
            except RiakError:
                print("+ %s (%s) search index checking operation is taken to queue." % (
                model.__name__, index_name))
                waiting_models.append(model)