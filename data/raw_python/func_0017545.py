def get_object(self, resource, object_type, content_ids, object_ids='*', location=0):
        """
        Get a list of Objects from a resource
        :param resource: The resource to get objects from
        :param object_type: The type of object to fetch
        :param content_ids: The unique id of the item to get objects for
        :param object_ids: ids of the objects to download
        :param location: The path to get Objects from
        :return: list
        """
        object_helper = GetObject()
        request_ids = object_helper.ids(content_ids=content_ids, object_ids=object_ids)

        response = self._request(
            capability='GetObject',
            options={
                'query':
                    {
                        "Resource": resource,
                        "Type": object_type,
                        "ID": ','.join(request_ids),
                        "Location": location
                    }
            }
        )

        if 'multipart' in response.headers.get('Content-Type'):
            parser = MultipleObjectParser()
            collection = parser.parse_image_response(response)
        else:
            parser = SingleObjectParser()
            collection = [parser.parse_image_response(response)]

        return collection