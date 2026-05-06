def parse_operations(self):
        """
        Flatten routes into a path -> method -> route structure
        """
        resource_defs = {
            getmeta(resources.Error).resource_name: resource_definition(resources.Error),
            getmeta(resources.Listing).resource_name: resource_definition(resources.Listing),
        }

        paths = collections.OrderedDict()
        for path, operation in self.parent.op_paths():
            # Cut of first item (will be the parents path)
            path = '/' + path[1:]  # type: UrlPath

            # Filter out swagger endpoints
            if self.SWAGGER_TAG in operation.tags:
                continue

            # Add to resource definitions
            if operation.resource:
                resource_defs[getmeta(operation.resource).resource_name] = resource_definition(operation.resource)

            # Add any resource definitions from responses
            if operation.responses:
                for response in operation.responses:
                    resource = response.resource
                    # Ensure we have a resource
                    if resource and resource is not DefaultResource:
                        resource_name = getmeta(resource).resource_name
                        # Don't generate a resource definition if one has already been created.
                        if resource_name not in resource_defs:
                            resource_defs[resource_name] = resource_definition(resource)

            # Add path parameters
            path_spec = paths.setdefault(path.format(self.swagger_node_formatter), {})

            # Add parameters
            parameters = self.generate_parameters(path)
            if parameters:
                path_spec['parameters'] = parameters

            # Add methods
            for method in operation.methods:
                path_spec[method.value.lower()] = operation.to_swagger()

        return paths, resource_defs