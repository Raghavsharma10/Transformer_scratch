def step_impl13(context, runs):
    """Check called apps / files.

    :param runs: expected number of records.
    :param context: test context.
    """
    executor_ = context.fuzz_executor
    stats = executor_.stats
    count = stats.cumulated_counts()
    assert count == runs, "VERIFY: Number of recorded runs."
    successful_runs = stats.cumulated_counts_for_status(Status.SUCCESS)
    assert successful_runs == runs