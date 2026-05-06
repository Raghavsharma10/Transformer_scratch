def merge(self):
        """Try merging all the bravado_core models across all loaded APIs. If
        duplicates occur, use the same bravado-core model to represent each, so
        bravado-core won't treat them as different models when passing them
        from one PyMacaron client stub to an other or when returning them via the
        PyMacaron server stub.
        """

        # The sole purpose of this method is to trick isinstance to return true
        # on model_values of the same kind but different apis/specs at:
        # https://github.com/Yelp/bravado-core/blob/4840a6e374611bb917226157b5948ee263913abc/bravado_core/marshal.py#L160

        log.info("Merging models of apis " + ", ".join(apis.keys()))

        # model_name => (api_name, model_json_def, bravado_core.model.MODELNAME)
        models = {}

        # First pass: find duplicate and keep only one model of each (fail if
        # duplicates have same name but different definitions)
        for api_name, api in apis.items():
            for model_name, model_def in api.api_spec.swagger_dict['definitions'].items():
                if model_name in models:
                    other_api_name, other_model_def, _ = models.get(model_name)
                    log.debug("Model %s in %s is a duplicate of one in %s" % (model_name, api_name, other_api_name))

                    if ApiPool._cmp_models(model_def, other_model_def) != 0:
                        raise MergeApisException("Cannot merge apis! Model %s exists in apis %s and %s but have different definitions:\n[%s]\n[%s]"
                                                 % (model_name, api_name, other_api_name, pprint.pformat(model_def), pprint.pformat(other_model_def)))
                else:
                    models[model_name] = (api_name, model_def, api.api_spec.definitions[model_name])

        # Second pass: patch every models and replace with the one we decided
        # to keep
        log.debug("Patching api definitions to remove all duplicates")
        for api_name, api in apis.items():
            for model_name in api.api_spec.definitions.keys():
                _, _, model_class = models.get(model_name)
                api.api_spec.definitions[model_name] = model_class