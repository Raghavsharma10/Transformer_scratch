def get_action_fields(self, view, action_name, resource):
        """
        Get fields exposed by action's serializer
        """
        serializer = view.get_serializer(resource)
        fields = OrderedDict()
        if not isinstance(serializer, view.serializer_class) or action_name == 'update':
            fields = self.get_fields(serializer.fields)
        return fields