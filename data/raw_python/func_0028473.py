def dealias_image(alias):
    '''Remove an image alias.'''
    with Session() as session:
        try:
            result = session.Image.dealiasImage(alias)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if result['ok']:
            print("alias {0} removed.".format(alias))
        else:
            print(result['msg'])