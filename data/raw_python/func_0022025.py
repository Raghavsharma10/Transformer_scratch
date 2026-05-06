def layers_to_solr(self, layers):
        """
        Sync n layers in Solr.
        """

        layers_dict_list = []
        layers_success_ids = []
        layers_errors_ids = []

        for layer in layers:
            layer_dict, message = layer2dict(layer)
            if not layer_dict:
                layers_errors_ids.append([layer.id, message])
                LOGGER.error(message)
            else:
                layers_dict_list.append(layer_dict)
                layers_success_ids.append(layer.id)

        layers_json = json.dumps(layers_dict_list)
        try:
            url_solr_update = '%s/solr/hypermap/update/json/docs' % SEARCH_URL
            headers = {"content-type": "application/json"}
            params = {"commitWithin": 1500}
            requests.post(url_solr_update, data=layers_json, params=params, headers=headers)
            LOGGER.info('Solr synced for the given layers')
        except Exception:
            message = "Error saving solr records: %s" % sys.exc_info()[1]
            layers_errors_ids.append([-1, message])
            LOGGER.error(message)
            return False, layers_errors_ids

        return True, layers_errors_ids