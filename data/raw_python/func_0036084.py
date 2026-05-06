def delete(dataset):
    """Use this function to delete dataset by it's id."""
    
    config = ApiConfig()
    client = ApiClient(config.host, config.app_id, config.app_secret)
    client.check_correct_host()
    client.delete(dataset)
    return ('Dataset {} has been deleted successfully'.format(dataset))