def sample_loop(job, uuid_list, inputs):
  """
  Loops over the sample_ids (uuids) in the manifest, creating child jobs to process each
  """

  for uuid_rg in uuid_list:

    uuid_items = uuid_rg.split(',')
    uuid = uuid_items[0]
    rg_line = None
    if len(uuid_items) > 1:
        rg_line = uuid_items[1]

    job.addChildJobFn(static_dag, uuid, rg_line, inputs)