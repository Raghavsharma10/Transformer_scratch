def get_actions(self, request, view):
        """
        Return metadata for resource-specific actions,
        such as start, stop, unlink
        """
        metadata = OrderedDict()
        actions = self.get_resource_actions(view)

        resource = view.get_object()
        for action_name, action in actions.items():
            if action_name == 'update':
                view.request = clone_request(request, 'PUT')
            else:
                view.action = action_name

            data = ActionSerializer(action, action_name, request, view, resource)
            metadata[action_name] = data.serialize()
            if not metadata[action_name]['enabled']:
                continue
            fields = self.get_action_fields(view, action_name, resource)
            if not fields:
                metadata[action_name]['type'] = 'button'
            else:
                metadata[action_name]['type'] = 'form'
                metadata[action_name]['fields'] = fields

            view.action = None
            view.request = request

        return metadata