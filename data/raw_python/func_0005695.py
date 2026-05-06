def get_additional_rewards(api):
    """returns list of non-user rewards (potion, armoire, gear)"""
    c = get_content(api)
    tasks = [c[i] for i in ['potion', 'armoire']]
    tasks.extend(api.user.inventory.buy.get())
    for task in tasks:
        task['id'] = task['alias'] = task['key']
    return tasks