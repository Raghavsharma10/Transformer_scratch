def enable():
    """
    Enable all benchmarking.
    """
    Benchmark.enable = True
    ComparisonBenchmark.enable = True
    BenchmarkedFunction.enable = True
    BenchmarkedClass.enable = True