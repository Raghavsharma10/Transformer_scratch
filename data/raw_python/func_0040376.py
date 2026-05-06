def execCommand(Argv, collect_missing):
  r"""Executes the given task with parameters.
  """
  try:
    return _execCommand(Argv, collect_missing)

  except Exception as e:
    if Settings['errorHandler']:
      Settings['errorHandler'](e)

    if Settings['debug']:
      # #ToDo: Have an option to debug through stderr. The issue is, the way to make pdb.post_mortem, to use stderr, like pdb.set_trace is unknown.
      import pdb
      pdb.post_mortem(sys.exc_info()[2])

    if not Settings['silent']: # Debug, then log the trace.
      import traceback

      etype, value, tb = sys.exc_info()
      tb = tb.tb_next.tb_next # remove the ec - calls from the traceback, to make it more understandable

      message = ''.join(traceback.format_exception(etype, value, tb))[:-1]

    else:
      if isinstance(e, HandledException): # let the modes handle the HandledException
        raise e

      message = str(e) # provide a succinct error message

    raise HandledException(message)