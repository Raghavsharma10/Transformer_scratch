def load_state_from_disk():
    """ loads the state from a local data.json file
    """
    if is_there_state():
        with open('data.json', 'r') as f:
            data = json.load(f)
        return data
    else:
        return False