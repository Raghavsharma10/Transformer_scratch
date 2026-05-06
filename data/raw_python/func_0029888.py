def update_milestone(id, **kwargs):
    """
    Update a ProductMilestone
    """
    data = update_milestone_raw(id, **kwargs)
    if data:
        return utils.format_json(data)