def _run_tool(job_name, t, fastafile, params):
    """Parallel motif prediction."""
    try:
        result = t.run(fastafile, params, mytmpdir())
    except Exception as e:
        result = ([], "", "{} failed to run: {}".format(job_name, e))
    
    return job_name, result