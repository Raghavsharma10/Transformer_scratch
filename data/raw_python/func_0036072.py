def cancel():
  """HTTP endpoint for canceling tasks

  If an active task is cancelled, an inactive task with the same code and the
  smallest interval will be activated if it exists.
  """
  task_id = request.form['id']
  task = Task.query.get(task_id)

  if not task:
    return json.dumps({
      'status': 'success',
      'id': None,
    })

  task.delete()

  if task.active:
    current_app.scheduler.cancel(task_id)

    code = task.code
    other_task = Task.query.filter_by(code=code).order_by('interval').first()
    if other_task:
      other_task.active = True
      other_task.save()
      current_app.scheduler.schedule({
        'id': other_task.id,
        'code': other_task.code,
        'interval': other_task.interval
      })

  return json.dumps({
    'status': 'success',
    'id': task_id,
  })