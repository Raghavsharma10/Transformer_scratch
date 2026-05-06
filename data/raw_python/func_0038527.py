def _analyze_case(model_dir, bench_dir, config):
    """ Generates statistics from the timing summaries """
    model_timings = set(glob.glob(os.path.join(model_dir, "*" + config["timing_ext"])))
    if bench_dir is not None:
        bench_timings = set(glob.glob(os.path.join(bench_dir, "*" + config["timing_ext"])))
    else:
        bench_timings = set()
    if not len(model_timings):
        return dict()
    model_stats = generate_timing_stats(model_timings, config['timing_vars'])
    bench_stats = generate_timing_stats(bench_timings, config['timing_vars'])
    return dict(model=model_stats, bench=bench_stats)