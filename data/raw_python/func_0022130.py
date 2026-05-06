def create_indices(catalog_slug):
        """Create ES core indices """
        # TODO: enable auto_create_index in the ES nodes to make this implicit.
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-index_.html#index-creation
        # http://support.searchly.com/customer/en/portal/questions/
        # 16312889-is-automatic-index-creation-disabled-?new=16312889
        mapping = {
            "mappings": {
                "layer": {
                    "properties": {
                        "layer_geoshape": {
                           "type": "geo_shape",
                           "tree": "quadtree",
                           "precision": REGISTRY_MAPPING_PRECISION
                        }
                    }
                }
            }
        }
        ESHypermap.es.indices.create(catalog_slug, ignore=[400, 404], body=mapping)