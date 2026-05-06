def add_atomic_information(data_api, data_setters):
    """Add all the structural information.
    :param data_api the data api from where to get the data
    :param data_setters the class to push the data to"""
    for model_chains in data_api.chains_per_model:
        data_setters.set_model_info(data_api.model_counter, model_chains)
        tot_chains_this_model = data_api.chain_counter + model_chains
        last_chain_counter = data_api.chain_counter
        for chain_index in range(last_chain_counter, tot_chains_this_model):
            add_chain_info(data_api, data_setters, chain_index)
        data_api.model_counter+=1