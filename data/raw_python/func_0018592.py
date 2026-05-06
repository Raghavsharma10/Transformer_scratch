def get_file(path=None, content=None):
    """
    :param path: relative path, or None to get from request
    :param content: file content, output in data. Used for editfile
    """
    if path is None:
        path = request.args.get('path')

    if path is None:
        return error('No path in request')
    
    filename = os.path.split(path.rstrip('/'))[-1]
    extension = filename.rsplit('.', 1)[-1]
    os_file_path = web_path_to_os_path(path)

    if os.path.isdir(os_file_path):
        file_type = 'folder'
        # Ensure trailing slash
        if path[-1] != '/':
            path += '/'
    else:
        file_type = 'file'

    ctime = int(os.path.getctime(os_file_path))
    mtime = int(os.path.getmtime(os_file_path))

    height = 0
    width = 0
    if extension in ['gif', 'jpg', 'jpeg', 'png']:
        try:
            im = PIL.Image.open(os_file_path)
            height, width = im.size
        except OSError:
            log.exception('Error loading image "{}" to get width and height'.format(os_file_path))
    
    attributes = {
        'name': filename,
        'path': get_url_path(path),
        'readable': 1 if os.access(os_file_path, os.R_OK) else 0,
        'writeable': 1 if os.access(os_file_path, os.W_OK) else 0,
        'created': datetime.datetime.fromtimestamp(ctime).ctime(),
        'modified': datetime.datetime.fromtimestamp(mtime).ctime(),
        'timestamp': mtime,
        'width': width,
        'height': height,
        'size': os.path.getsize(os_file_path)
    }

    if content:
        attributes['content'] = content

    return {
        'id': path,
        'type': file_type,
        'attributes': attributes
    }