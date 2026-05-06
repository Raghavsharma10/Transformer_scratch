def run(run_type, module, config):
    """
    Collects the analyses cases to be run and launches processes for each of
    them.

    Args:
        run_type: A string representation of the run type (eg. verification)
        module: The module corresponding to the run.  Must have a run_suite function
        config: The configuration for the module
    """
    print(" -----------------------------------------------------------------")
    print("   Beginning " + run_type.lower() + " test suite ")
    print(" -----------------------------------------------------------------")
    print("")
    summary = run_quiet(module, config)
    print(" -----------------------------------------------------------------")
    print("   " + run_type.capitalize() + " test suite complete ")
    print(" -----------------------------------------------------------------")
    print("")
    return summary