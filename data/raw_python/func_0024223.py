def apply_change(self):
        """Applies changes to the permissions of the role.

        To make a change to the permission of the role, a request
        in the following format should be sent:

        .. code-block:: python
            {
                'change':
                    {
                        'id': 'workflow2.lane1.task1',
                        'checked': false
                    },
            }

        The 'id' field of the change is the id of the tree element
        that was sent to the UI (see `Permissions.edit_permissions`).
        'checked' field is the new state of the element.
        """
        changes = self.input['change']
        key = self.current.task_data['role_id']
        role = RoleModel.objects.get(key=key)
        for change in changes:
            permission = PermissionModel.objects.get(code=change['id'])
            if change['checked'] is True:
                role.add_permission(permission)
            else:
                role.remove_permission(permission)
        role.save()