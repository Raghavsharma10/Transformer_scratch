def qdel(jobid, context='grid'):
  """Halts a given job.

  Keyword parameters:

  jobid
    The job identifier as returned by qsub()

  context
    The setshell context in which we should try a 'qsub'. Normally you don't
    need to change the default. This variable can also be set to a context
    dictionary in which case we just setup using that context instead of
    probing for a new one, what can be fast.
  """

  scmd = ['qdel', '%d' % jobid]

  logger.debug("Qdel command '%s'", ' '.join(scmd))

  from .setshell import sexec
  sexec(context, scmd, error_on_nonzero=False)