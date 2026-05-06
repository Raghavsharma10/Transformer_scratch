def clear_es():
        """Clear all indexes in the es core"""
        # TODO: should receive a catalog slug.
        ESHypermap.es.indices.delete(ESHypermap.index_name, ignore=[400, 404])
        LOGGER.debug('Elasticsearch: Index cleared')