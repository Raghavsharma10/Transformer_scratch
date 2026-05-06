def to_query(self):
        """
        Returns a json-serializable representation.
        """
        return {
            "geo_shape": {
                self.name: {
                    "indexed_shape":  {
                        "index": self.index_name,
                        "type": self.doc_type,
                        "id": self.shape_id,
                        "path": self.path
                    }
                }
            }
        }