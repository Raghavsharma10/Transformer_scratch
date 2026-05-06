def file_put_contents(path, data):
    """ Put passed contents into file located at 'path' """
    with open(path, 'w') as f:
        f.write(data); f.flush()