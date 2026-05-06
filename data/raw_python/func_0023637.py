def do_ls(client, args):
    """List directory"""

    for item in client.get_folder_contents_iter(args.uri):
        # privacy flag
        if item['privacy'] == 'public':
            item['pf'] = '@'
        else:
            item['pf'] = '-'

        if isinstance(item, Folder):
            # type flag
            item['tf'] = 'd'
            item['key'] = item['folderkey']
            item['size'] = ''
        else:
            item['tf'] = '-'
            item['key'] = item['quickkey']
            item['name'] = item['filename']

        print("{tf}{pf} {key:>15} {size:>10} {created} {name}".format(**item))

    return True