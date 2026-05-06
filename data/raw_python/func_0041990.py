def qstat(jobid, context='grid'):
  """Queries status of a given job.

  Keyword parameters:

  jobid
    The job identifier as returned by qsub()

  context
    The setshell context in which we should try a 'qsub'. Normally you don't
    need to change the default. This variable can also be set to a context
    dictionary in which case we just setup using that context instead of
    probing for a new one, what can be fast.

  Returns a dictionary with the specific job properties
  """

  scmd = ['qstat', '-j', '%d' % jobid, '-f']

  logger.debug("Qstat command '%s'", ' '.join(scmd))

  from .setshell import sexec
  data = str_(sexec(context, scmd, error_on_nonzero=False))

  # some parsing:
  retval = {}
  for line in data.split('\n'):
    s = line.strip()
    if s.lower().find('do not exist') != -1: return {}
    if not s or s.find(10*'=') != -1: continue
    kv = QSTAT_FIELD_SEPARATOR.split(s, 1)
    if len(kv) == 2: retval[kv[0]] = kv[1]

  return retval