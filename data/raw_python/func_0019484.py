def pp_predict_motifs(fastafile, outfile, analysis="small", organism="hg18", single=False, background="", tools=None, job_server=None, ncpus=8, max_time=-1, stats_fg=None, stats_bg=None):
    """Parallel prediction of motifs.

    Utility function for gimmemotifs.denovo.gimme_motifs. Probably better to 
    use that, instead of this function directly.
    """
    if tools is None:
        tools = {}

    config = MotifConfig()

    if not tools:
        tools = dict([(x,1) for x in config.get_default_params["tools"].split(",")])
    
    #logger = logging.getLogger('gimme.prediction.pp_predict_motifs')

    wmin = 5 
    step = 1
    if analysis in ["large","xl"]:
        step = 2
        wmin = 6
    
    analysis_max = {"xs":5,"small":8, "medium":10,"large":14, "xl":20}
    wmax = analysis_max[analysis]

    if analysis == "xs":
        sys.stderr.write("Setting analysis xs to small")
        analysis = "small"

    
    if not job_server:
        n_cpus = int(config.get_default_params()["ncpus"])
        job_server = Pool(processes=n_cpus, maxtasksperchild=1000) 
    
    jobs = {}
    
    result = PredictionResult(
                outfile, 
                fg_file=stats_fg, 
                background=stats_bg,
                job_server=job_server,
                )
    
    # Dynamically load all tools
    toolio = [x[1]() for x in inspect.getmembers(
                                                tool_classes, 
                                                lambda x: 
                                                        inspect.isclass(x) and 
                                                        issubclass(x, tool_classes.MotifProgram)
                                                ) if x[0] != 'MotifProgram']
    
    # TODO:
    # Add warnings for running time: Weeder, GADEM
        
    ### Add all jobs to the job_server ###
    params = {
            'analysis': analysis, 
            'background':background, 
            "single":single, 
            "organism":organism
            }
    
    # Tools that don't use a specified width usually take longer
    # ie. GADEM, XXmotif, MEME
    # Start these first.
    for t in [tool for tool in toolio if not tool.use_width]:
        if t.name in tools and tools[t.name]:
            logger.debug("Starting %s job", t.name)
            job_name = t.name
            jobs[job_name] = job_server.apply_async(
                        _run_tool,
                        (job_name, t, fastafile, params), 
                        callback=result.add_motifs)
        else:
            logger.debug("Skipping %s", t.name)

    for t in [tool for tool in toolio if tool.use_width]:
        if t.name in tools and tools[t.name]:
            for i in range(wmin, wmax + 1, step):
                logger.debug("Starting %s job, width %s", t.name, i)
                job_name = "%s_width_%s" % (t.name, i)
                my_params = params.copy()
                my_params['width'] = i
                jobs[job_name] = job_server.apply_async(
                    _run_tool,
                    (job_name, t, fastafile, my_params), 
                    callback=result.add_motifs)
        else:
            logger.debug("Skipping %s", t.name)
    
    logger.info("all jobs submitted")
    for job in jobs.values():
        job.get()

    result.wait_for_stats()
    ### Wait until all jobs are finished or the time runs out ###
#    start_time = time()    
#    try:
#        # Run until all jobs are finished
#        while len(result.finished) < len(jobs.keys()) and (not(max_time) or time() - start_time < max_time):
#            pass
#        if len(result.finished) < len(jobs.keys()):
#            logger.info("Maximum allowed running time reached, destroying remaining jobs")
#            job_server.terminate()
#            result.submit_remaining_stats()
#    ### Or the user gets impatient... ###
#    except KeyboardInterrupt:
#        # Destroy all running jobs
#        logger.info("Caught interrupt, destroying all running jobs")
#        job_server.terminate()
#        result.submit_remaining_stats()
#        
#    
#    if stats_fg and stats_bg:
#        logger.info("waiting for motif statistics")
#        n = 0
#        last_len = 0 
#       
#    
#        while len(set(result.stats.keys())) < len(set([str(m) for m in result.motifs])):
#            if n >= 30:
#                logger.debug("waited long enough")
#                logger.debug("motifs: %s, stats: %s", len(result.motifs), len(result.stats.keys()))
#                for i,motif in enumerate(result.motifs):
#                    if "{}_{}".format(motif.id, motif.to_consensus()) not in result.stats:
#                        logger.debug("deleting %s", motif)
#                        del result.motifs[i]
#                break
#            sleep(2)
#            if len(result.stats.keys()) == last_len:
#                n += 1
#            else:
#                last_len = len(result.stats.keys())
#                n = 0
#    
    return result