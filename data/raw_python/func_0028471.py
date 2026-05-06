def rescan_images(registry):
    '''Update the kernel image metadata from all configured docker registries.'''
    with Session() as session:
        try:
            result = session.Image.rescanImages(registry)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if result['ok']:
            print("kernel image metadata updated")
        else:
            print("rescanning failed: {0}".format(result['msg']))