def airline_delay(data_set='airline_delay', num_train=700000, num_test=100000, seed=default_seed):
    """Airline delay data used in Gaussian Processes for Big Data by Hensman, Fusi and Lawrence"""

    if not data_available(data_set):
        download_data(data_set)

    dir_path = os.path.join(data_path, data_set)
    filename = os.path.join(dir_path, 'filtered_data.pickle')

    # 1. Load the dataset
    import pandas as pd
    data = pd.read_pickle(filename)

    # WARNING: removing year
    data.pop('Year')

    # Get data matrices
    Yall = data.pop('ArrDelay').values[:,None]
    Xall = data.values

    # Subset the data (memory!!)
    all_data = num_train+num_test
    Xall = Xall[:all_data]
    Yall = Yall[:all_data]

    # Get testing points
    np.random.seed(seed=seed)
    N_shuffled = permute(Yall.shape[0])
    train, test = N_shuffled[num_test:], N_shuffled[:num_test]
    X, Y = Xall[train], Yall[train]
    Xtest, Ytest = Xall[test], Yall[test]
    covariates =  ['month', 'day of month', 'day of week', 'departure time', 'arrival time', 'air time', 'distance to travel', 'age of aircraft / years']
    response = ['delay']
    return data_details_return({'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'seed' : seed, 'info': "Airline delay data used for demonstrating Gaussian processes for big data.", 'covariates': covariates, 'response': response}, data_set)