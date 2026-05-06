def get_task_actions(current):
    """
           List task types for current user


           .. code-block:: python

               #  request:
                   {
                   'view': '_zops_get_task_actions',
                   'key': key,
                   }

               #  response:
                   {
                   'key': key,
                   'actions': [{"title":':'Action Title', "wf": "workflow_name"},]
                    }
    """
    task_inv = TaskInvitation.objects.get(current.input['key'])
    actions = [{"title": __(u"Assign Someone Else"), "wf": "assign_same_abstract_role"},
               {"title": __(u"Suspend"), "wf": "suspend_workflow"},
               {"title": __(u"Postpone"), "wf": "postpone_workflow"}]
    if task_inv.instance.current_actor != current.role:
        actions.append({"title": __(u"Assign Yourself"), "wf": "task_assign_yourself"})

    current.output['key'] = task_inv.key
    current.output['actions'] = actions