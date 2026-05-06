def disable_precompute(panel):
  """Cancel precomputation for `panel`"""
  task_id = panel['data_source']['precompute']['task_id']
  result = scheduler_client.cancel(task_id)
  if result['status'] != 'success':
    raise RuntimeError(result.get('reason'))