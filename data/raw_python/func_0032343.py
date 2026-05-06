def split_resource_path(resource):
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in resource.split('/'):
        if path.sep in piece \
           or (path.altsep and path.altsep in piece) or \
           piece == path.pardir:
            raise ResourceNotFound(resource)
        elif piece and piece != '.':
            pieces.append(piece)
    return pieces