def load_network_model(model):
    '''
    Loads metabolic network models in metabolitics.

    :param str model: model name
    '''
    if type(model) == str:
        if model in ['ecoli', 'textbook', 'salmonella']:
            return cb.test.create_test_model(model)
        elif model == 'recon2':
            return cb.io.load_json_model('%s/network_models/%s.json' %
                                         (DATASET_PATH, model))
    if type(model) == cb.Model:
        return model