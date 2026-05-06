def get_task_detail(current):
    """
           Show task details

           .. code-block:: python

               #  request:
                   {
                   'view': '_zops_get_task_detail',
                   'key': key,
                   }

               #  response:
                   {
                   'task_title': string,
                   'task_detail': string, # markdown formatted text
                    }
    """
    task_inv = TaskInvitation.objects.get(current.input['key'])
    obj = task_inv.instance.get_object()
    current.output['task_title'] = task_inv.instance.task.name
    current.output['task_detail'] = """Explain: %s
    State: %s""" % (obj.__unicode__() if obj else '', task_inv.progress)