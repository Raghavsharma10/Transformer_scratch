def inject_context(context):
    """
    Updates context.json with data from JSON-string given as param.
    :param context:
    :return:
    """
    context_path = tasks.get_context_path()
    try:
        new_context = json.loads(context)
    except ValueError:
        print('Couldn\'t load context parameter')
        return
    with open(context_path) as jsonfile:
        try:
            jsondata = json.loads(jsonfile.read())
            jsondata.update(new_context)
        except ValueError:
            print('Couldn\'t read context.json')
            return
    with open(context_path, 'w') as jsoncontent:
        json.dump(jsondata, jsoncontent, indent=4)