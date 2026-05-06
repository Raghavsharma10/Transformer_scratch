def run_competition(builders=[], task=BalanceTask(), Optimizer=HillClimber, rounds=3, max_eval=20, N_hidden=3, verbosity=0):
    """ pybrain buildNetwork builds a subtly different network structhan build_ann... so compete them!

    Arguments:
        task (Task): task to compete at
        Optimizer (class): pybrain.Optimizer class to instantiate for each competitor
        rounds (int): number of times to run the competition
        max_eval (int): number of objective function evaluations that the optimizer is allowed
          in each round
        N_hidden (int): number of hidden nodes in each network being competed

    The functional difference that I can see is that:

      buildNetwork connects the bias to the output
      build_ann does not

    The api differences are:

      build_ann allows heterogeneous layer types but the output layer is always linear
      buildNetwork allows specification of the output layer type

    """
    results = []
    builders = list(builders) + [buildNetwork, util.build_ann]

    for r in range(rounds):
        heat = []

        # FIXME: shuffle the order of the builders to keep things fair
        #        (like switching sides of the tennis court)
        for builder in builders:
            try:
                competitor = builder(task.outdim, N_hidden, task.indim, verbosity=verbosity)
            except NetworkError:
                competitor = builder(task.outdim, N_hidden, task.indim)

            # TODO: verify that a full reset is actually happening
            task.reset()
            optimizer = Optimizer(task, competitor, maxEvaluations=max_eval)
            t0 = time.time()
            nn, nn_best = optimizer.learn()
            t1 = time.time()
            heat += [(nn_best, t1-t0, nn)]
        results += [tuple(heat)]
        if verbosity >= 0:
            print([competitor_scores[:2] for competitor_scores in heat])

    # # alternatively:
    # agent = ( pybrain.rl.agents.OptimizationAgent(net, HillClimber())
    #             or
    #           pybrain.rl.agents.LearningAgent(net, pybrain.rl.learners.ENAC()) )
    # exp = pybrain.rl.experiments.EpisodicExperiment(task, agent).doEpisodes(100)

    means = [[np.array([r[i][j] for r in results]).mean() for i in range(len(results[0]))] for j in range(2)]
    if verbosity > -1:
        print('Mean Performance:')
        print(means)
        perfi, speedi = np.argmax(means[0]), np.argmin(means[1])
        print('And the winner for performance is ... Algorithm #{} (0-offset array index [{}])'.format(perfi+1, perfi))
        print('And the winner for speed is ...       Algorithm #{} (0-offset array index [{}])'.format(speedi+1, speedi))

    return results, means