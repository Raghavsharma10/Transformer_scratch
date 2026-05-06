def print_result(overview, *names):
    """ Print the result of a verisure request """
    if names:
        for name in names:
            toprint = overview
            for part in name.split('/'):
                toprint = toprint[part]
            print(json.dumps(toprint, indent=4, separators=(',', ': ')))
    else:
        print(json.dumps(overview, indent=4, separators=(',', ': ')))