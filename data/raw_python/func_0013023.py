def format_tasks(tasks):
  """Converts a list of tasks to a list of string representations.

  Args:
    tasks: A list of the tasks to convert.
  Returns:
    A list of string formatted tasks.
  """
  return ['%d : %s (%s)' % (task.key.id(),
                            task.description,
                            ('done' if task.done
                             else 'created %s' % task.created))
          for task in tasks]