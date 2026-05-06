def get_roc_values(motif, fg_file, bg_file):
    """Calculate ROC AUC values for ROC plots."""
    #print(calc_stats(motif, fg_file, bg_file, stats=["roc_values"], ncpus=1))
    #["roc_values"])
    
    try:
#        fg_result = motif.pwm_scan_score(Fasta(fg_file), cutoff=0.0, nreport=1)
#        fg_vals = [sorted(x)[-1] for x in fg_result.values()]
#
#        bg_result = motif.pwm_scan_score(Fasta(bg_file), cutoff=0.0, nreport=1)
#        bg_vals = [sorted(x)[-1] for x in bg_result.values()]

#        (x, y) = roc_values(fg_vals, bg_vals)
        stats = calc_stats(motif, fg_file, bg_file, stats=["roc_values"], ncpus=1)
        (x,y) = list(stats.values())[0]["roc_values"]
        return None,x,y
    except Exception as e:
        print(motif)
        print(motif.id)
        raise
        error = e
        return error,[],[]