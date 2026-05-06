def from_line(cls, name, comes_from=None):
        """Creates an InstallRequirement from a name, which might be a
        requirement, directory containing 'setup.py', filename, or URL.
        """
        url = None
        name = name.strip()
        req = None
        path = os.path.normpath(os.path.abspath(name))
        link = None

        if is_url(name):
            link = Link(name)
        elif os.path.isdir(path) and (os.path.sep in name or name.startswith('.')):
            if not is_installable_dir(path):
                raise InstallationError("Directory %r is not installable. File 'setup.py' not found.", name)
            link = Link(path_to_url(name))
        elif is_archive_file(path):
            if not os.path.isfile(path):
                logger.warn('Requirement %r looks like a filename, but the file does not exist', name)
            link = Link(path_to_url(name))

        # If the line has an egg= definition, but isn't editable, pull the requirement out.
        # Otherwise, assume the name is the req for the non URL/path/archive case.
        if link and req is None:
          url = link.url_fragment
          req = link.egg_fragment

          # Handle relative file URLs
          if link.scheme == 'file' and re.search(r'\.\./', url):
            url = path_to_url(os.path.normpath(os.path.abspath(link.path)))

        else:
          req = name

        return cls(req, comes_from, url=url)