def determine_2(self, container_name, container_alias, meta, val):
        """"Default the alias to the name of the container"""
        if container_alias is not NotSpecified:
            return container_alias
        return container_name[container_name.rfind(":")+1:].replace('/', '-')