def get_task_types(current):
    """
           List task types for current user


           .. code-block:: python

               #  request:
                   {
                   'view': '_zops_get_task_types',
                   }

               #  response:
                   {
                   'task_types': [
                       {'name': string, # wf name
                        'title': string,  # title of workflow
                        },]
                        }
    """
    current.output['task_types'] = [{'name': bpmn_wf.name,
                                     'title': bpmn_wf.title}
                                    for bpmn_wf in BPMNWorkflow.objects.all()
                                    if current.has_permission(bpmn_wf.name)]