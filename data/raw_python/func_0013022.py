def mark_done(task_id):
  """Marks a task as done.

  Args:
    task_id: The integer id of the task to update.

  Raises:
    ValueError: if the requested task doesn't exist.
  """
  task = Task.get_by_id(task_id)
  if task is None:
    raise ValueError('Task with id %d does not exist' % task_id)
  task.done = True
  task.put()