def step_impl12(context, runs):
    """Check called apps / files.

    :param runs: expected number of records.
    :param context: test context.
    """
    executor_ = context.fuzz_executor
    stats = executor_.stats
    count = stats.cumulated_counts()
    assert count == runs, "VERIFY: Number of recorded runs."