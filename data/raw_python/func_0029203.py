def json2table(json):
    """This does format a dictionary into a table.
    Note this expects a dictionary (not a json string!)

    :param json:
    :return:
    """
    filter_terms = ['ResponseMetadata']
    table = []
    try:
        for k in filter(lambda k: k not in filter_terms, json.keys()):
            table.append([k.encode('ascii', 'ignore'),
                         str(json[k]).encode('ascii', 'ignore')])
        return tabulate(table, tablefmt='fancy_grid')
    except GracefulExit:
        raise
    except Exception as e:
        log.error(e)
        return json