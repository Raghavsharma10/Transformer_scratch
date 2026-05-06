def _parse_response(response, clazz, is_list=False, resource_name=None):
        """Parse a Marathon response into an object or list of objects."""
        target = response.json()[
            resource_name] if resource_name else response.json()
        if is_list:
            return [clazz.from_json(resource) for resource in target]
        else:
            return clazz.from_json(target)