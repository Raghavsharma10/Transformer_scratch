def disable():
    """
    Disable all benchmarking.
    """
    Benchmark.enable = False
    ComparisonBenchmark.enable = False
    BenchmarkedFunction.enable = False
    BenchmarkedClass.enable = False