def find_models_and_delete_search_index(self, model, force, exec_models, check_only):
        """
        Finds models to execute and these models' exist search indexes are deleted.
        For other operations, necessary models are gathered to list(exec_models)

        Args:
            model: model to execute
            force(bool): True or False if True, all given models are executed.
            exec_models(list): if not force, models to execute are gathered to list.
            If there is not necessity to migrate operation model doesn't put to exec list.
            check_only: do not migrate, only report migration is needed or not if True

        Returns:

        """
        ins = model(fake_context)
        fields = self.get_schema_fields(ins._collect_index_fields())
        new_schema = self.compile_schema(fields)
        bucket_name = model._get_bucket_name()
        bucket_type = client.bucket_type(settings.DEFAULT_BUCKET_TYPE)
        bucket = bucket_type.bucket(bucket_name)
        index_name = "%s_%s" % (settings.DEFAULT_BUCKET_TYPE, bucket_name)
        if not force:
            try:
                schema = get_schema_from_solr(index_name)
                if schema == new_schema:
                    print("Schema %s is already up to date, nothing to do!" % index_name)
                    return
                elif check_only and schema != new_schema:
                    print("Schema %s is not up to date, migrate this model!" % index_name)
                    return
            except:
                import traceback
                traceback.print_exc()
        bucket.set_property('search_index', 'foo_index')
        try:
            client.delete_search_index(index_name)
        except RiakError as e:
            if 'notfound' != e.value:
                raise
        wait_for_schema_deletion(index_name)
        exec_models.append(model)