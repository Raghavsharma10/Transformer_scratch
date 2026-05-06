def generate_xhtml(path, dirs, files):
    """Return a XHTML document listing the directories and files."""
    # Prepare the path to display.
    if path != '/':
        dirs.insert(0, '..')
    if not path.endswith('/'):
        path += '/'

    def itemize(item):
        return '<a href="%s">%s</a>' % (item, path+item)
    dirs = [d + '/' for d in dirs]
    return """
    <html>
     <body>
      <h1>%s</h1>
       <pre>%s\n%s</pre>
     </body>
    </html>
    """ % (path, '\n'.join(itemize(dir) for dir in dirs), '\n'.join(itemize(file) for file in files))