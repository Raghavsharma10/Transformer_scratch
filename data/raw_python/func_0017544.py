def get_preferred_object(self, resource, object_type, content_id, location=0):
        """
        Get the first object from a Resource
        :param resource: The name of the resource
        :param object_type: The type of object to fetch
        :param content_id: The unique id of the item to get objects for
        :param location: The path to get Objects from
        :return: Object
        """
        collection = self.get_object(resource=resource, object_type=object_type,
                                     content_ids=content_id, object_ids='0', location=location)
        return collection[0]