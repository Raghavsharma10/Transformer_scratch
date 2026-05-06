def get_experiment_results():
    """
    Computes the results of all experiments, stores it in redis, and prints it
    out
    """

    redis = oz.redis.create_connection()

    for experiment in oz.bandit.get_experiments(redis):
        experiment.compute_default_choice()
        csq, confident = experiment.confidence()

        print("%s:" % experiment.name)
        print("- creation date: %s" % experiment.metadata["creation_date"])
        print("- default choice: %s" % experiment.default_choice)
        print("- chi squared: %s" % csq)
        print("- confident: %s" % confident)
        print("- choices:")

        for choice in experiment.choices:
            print("  - %s: plays=%s, rewards=%s, performance=%s" % (choice.name, choice.plays, choice.rewards, choice.performance))