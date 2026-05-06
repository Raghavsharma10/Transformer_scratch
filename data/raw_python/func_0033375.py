def sync_from_spec(redis, schema):
    """
    Takes an input experiment spec and creates/modifies/archives the existing
    experiments to match the spec.

    If there's an experiment in the spec that currently doesn't exist, it will
    be created along with the associated choices.

    If there's an experiment in the spec that currently exists, and the set of
    choices are different, that experiment's choices will be modified to match
    the spec.

    If there's an experiment not in the spec that currently exists, it will be
    archived.

    A spec looks like this:

    {
       "experiment 1": ["choice 1", "choice 2", "choice 3"],
       "experiment 2": ["choice 1", "choice 2"]
    }
    """

    def get_experiments_dict(active=True):
        """Returns a dictionary of experiment names -> experiment objects"""
        return dict((experiment.name, experiment) for experiment in get_experiments(redis, active=active))

    # Get the current experiments
    active_experiments = get_experiments_dict()
    archived_experiments = get_experiments_dict(active=False)

    # Get the newly defined experiment names and the names of the experiments
    # already setup
    new_experiment_names = set(schema.keys())
    active_experiment_names = set(active_experiments.keys())

    # Find all the experiments that are in the schema and are defined among the
    # archived experiments, but not the active ones (we check against active
    # experiments to prevent the edge case where an experiment might be defined
    # doubly in both active and archived experiments)
    unarchivable_experiment_names = (new_experiment_names - active_experiment_names) & set(archived_experiments.keys())

    # De-archive the necessary experiments
    for unarchivable_experiment_name in unarchivable_experiment_names:
        print("- De-archiving %s" % unarchivable_experiment_name)

        # Because there is no function to de-archive an experiment, it must
        # be done manually
        pipe = redis.pipeline(transaction=True)
        pipe.sadd(ACTIVE_EXPERIMENTS_REDIS_KEY, unarchivable_experiment_name)
        pipe.srem(ARCHIVED_EXPERIMENTS_REDIS_KEY, unarchivable_experiment_name)
        pipe.execute()

    # Reload the active experiments if we de-archived any
    if unarchivable_experiment_names:
        active_experiments = get_experiments_dict()
        active_experiment_names = set(active_experiments.keys())

    # Create the new experiments
    for new_experiment_name in new_experiment_names - active_experiment_names:
        print("- Creating experiment %s" % new_experiment_name)

        experiment = add_experiment(redis, new_experiment_name)

        for choice in schema[new_experiment_name]:
            print("  - Adding choice %s" % choice)
            experiment.add_choice(choice)

    # Archive experiments not defined in the schema
    for archivable_experiment_name in active_experiment_names - new_experiment_names:
        print("- Archiving %s" % archivable_experiment_name)
        active_experiments[archivable_experiment_name].archive()

    # Update the choices for existing experiments that are also defined in the
    # schema
    for experiment_name in new_experiment_names & active_experiment_names:
        experiment = active_experiments[experiment_name]
        new_choice_names = set(schema[experiment_name])
        old_choice_names = set(experiment.choice_names)

        # Add choices in the schema that do not exist yet
        for new_choice_name in new_choice_names - old_choice_names:
            print("- Adding choice %s to existing experiment %s" % (new_choice_name, experiment_name))
            experiment.add_choice(new_choice_name)

        # Remove choices that aren't in the schema
        for removable_choice_name in old_choice_names - new_choice_names:
            print("- Removing choice %s from existing experiment %s" % (removable_choice_name, experiment_name))
            experiment.remove_choice(removable_choice_name)