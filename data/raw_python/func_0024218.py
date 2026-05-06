def edit_permissions(self):
        """Creates the view used to edit permissions.

        To create the view, data in the following format is passed to the UI
        in the objects field:

        .. code-block:: python
            {
                "type": "tree-toggle",
                "action": "set_permission",
                "tree": [
                    {
                        "checked": true,
                        "name": "Workflow 1 Name",
                        "id": "workflow1",
                        "children":
                            [
                                {
                                    "checked": true,
                                    "name": "Task 1 Name",
                                    "id": "workflow1..task1",
                                    "children": []
                                },
                                {
                                    "checked": false,
                                    "id": "workflow1..task2",
                                    "name": "Task 2 Name",
                                    "children": []
                                }
                            ]
                    },
                    {
                        "checked": true,
                        "name": "Workflow 2 Name",
                        "id": "workflow2",
                        "children": [
                            {
                                "checked": true,
                                "name": "Workflow 2 Lane 1 Name",
                                "id": "workflow2.lane1",
                                "children": [
                                    {
                                        "checked": true,
                                        "name": "Workflow 2 Task 1 Name",
                                        "id": "workflow2.lane1.task1",
                                        "children": []
                                    },
                                    {
                                        "checked": false,
                                        "name": "Workflow 2 Task 2 Name",
                                        "id": "workflow2.lane1.task2",
                                        "children": []
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

        "type" field denotes that the object is a tree view which has elements that can be toggled.
        "action" field is the

        "name" field is the human readable name.
        "id" field is used to make requests to the backend.
        "checked" field shows whether the role has the permission or not.
        "children" field is the sub-permissions of the permission.
        """
        # Get the role that was selected in the CRUD view
        key = self.current.input['object_id']
        self.current.task_data['role_id'] = key
        role = RoleModel.objects.get(key=key)
        # Get the cached permission tree, or build a new one if there is none cached
        # TODO: Add an extra view in case there was no cache, as in 'please wait calculating permissions'
        permission_tree = self._permission_trees(PermissionModel.objects)
        # Apply the selected role to the permission tree, setting the 'checked' field
        # of the permission the role has
        role_tree = self._apply_role_tree(permission_tree, role)
        # Apply final formatting, and output the tree to the UI
        self.output['objects'] = [
            {
                'type': 'tree-toggle',
                'action': 'apply_change',
                'trees': self._format_tree_output(role_tree),
            },
        ]
        self.form_out(PermissionForm())