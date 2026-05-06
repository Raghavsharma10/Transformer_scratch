def _get_model(vehicle):
    """Clean the model field. Best guess."""
    model = vehicle['model']
    model = model.replace(vehicle['year'], '')
    model = model.replace(vehicle['make'], '')
    return model.strip().split(' ')[0]