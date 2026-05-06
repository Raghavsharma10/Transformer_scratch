def predict_motifs(infile, bgfile, outfile, params=None, stats_fg=None, stats_bg=None):
    """ Predict motifs, input is a FASTA-file"""

    # Parse parameters
    required_params = ["tools", "available_tools", "analysis", 
                                "genome", "use_strand", "max_time"]
    if params is None:
        params = parse_denovo_params()
    else:
        for p in required_params:
            if p not in params:
                params = parse_denovo_params()
                break
    
    # Define all tools
    tools = dict(
            [
                (x.strip(), x in [y.strip() for y in params["tools"].split(",")]) 
                    for x in params["available_tools"].split(",")
            ]
            )

    # Predict the motifs
    analysis = params["analysis"]
    logger.info("starting motif prediction (%s)", analysis)
    logger.info("tools: %s", 
            ", ".join([x for x in tools.keys() if tools[x]]))
    result = pp_predict_motifs(
                    infile, 
                    outfile, 
                    analysis, 
                    params.get("genome", None), 
                    params["use_strand"], 
                    bgfile, 
                    tools, 
                    None, 
                    #logger=logger, 
                    max_time=params["max_time"], 
                    stats_fg=stats_fg, 
                    stats_bg=stats_bg
                )

    motifs = result.motifs
    logger.info("predicted %s motifs", len(motifs))
    logger.debug("written to %s", outfile)

    if len(motifs) == 0:
        logger.info("no motifs found")
        result.motifs = []
    
    return result