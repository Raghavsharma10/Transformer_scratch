def step_impl11(context, runs):
    """Execute multiple runs.

    :param runs: number of test runs to perform.
    :param context: test context.
    """
    executor = context.fuzz_executor
    executor.run_test(runs)
    stats = executor.stats
    count = stats.cumulated_counts()
    assert count == runs, "VERIFY: stats available."