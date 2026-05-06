def delete_relationship(cls, id, related_collection_name, related_resource=None):
        """
        Deprecated for version 1.1.0.  Please use update_relationship
        """
        try:
            this_resource = cls.nodes.get(id=id, active=True)
            if not related_resource:
                r = this_resource.delete_relationship_collection(related_collection_name)
            else:
                r = this_resource.delete_individual_relationship(related_collection_name, related_resource)
        except DoesNotExist:
            r = application_codes.error_response([application_codes.RESOURCE_NOT_FOUND])
        return r