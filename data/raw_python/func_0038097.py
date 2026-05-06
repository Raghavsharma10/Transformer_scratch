def launch_processes(tests, run_module, group=True, **config):
    """ Helper method to launch processes and sync output """
    manager = multiprocessing.Manager()
    test_summaries = manager.dict()
    process_handles = [multiprocessing.Process(target=run_module.run_suite,
                       args=(test, config[test], test_summaries)) for test in tests]
    for p in process_handles:
        p.start()
    for p in process_handles:
        p.join()

    if group:
        summary = run_module.populate_metadata(tests[0], config[tests[0]])
        summary["Data"] = dict(test_summaries)
        return summary
    else:
        test_summaries = dict(test_summaries)
        summary = []
        for ii, test in enumerate(tests):
            summary.append(run_module.populate_metadata(test, config[test]))
            if summary[ii]:
                summary[ii]['Data'] = {test: test_summaries[test]}
        return summary