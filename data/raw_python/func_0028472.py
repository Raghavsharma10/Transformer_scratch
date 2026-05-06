def alias_image(alias, target):
    '''Add an image alias.'''
    with Session() as session:
        try:
            result = session.Image.aliasImage(alias, target)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if result['ok']:
            print("alias {0} created for target {1}".format(alias, target))
        else:
            print(result['msg'])