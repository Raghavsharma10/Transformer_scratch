def main():
    """Read configuration and execute test runs."""
    parser = argparse.ArgumentParser(description='Stress test applications.')
    parser.add_argument('config_path', help='Path to configuration file.')
    args = parser.parse_args()
    try:
        configuration = load_configuration(args.config_path)
    except InvalidConfigurationError:
        print("\nConfiguration is not valid.")
        print('Example:\n{}'.format(help_configuration))
        return 1
    print("Starting up ...")
    futures = []
    with ProcessPoolExecutor(configuration[PROCESSORS]) as executor:
        for _ in range(configuration[PROCESSES]):
            futures.append(executor.submit(execute_test, configuration))
    print("... finished")
    test_stats = combine_test_stats([f.result() for f in futures])
    show_test_stats(test_stats)
    return 0