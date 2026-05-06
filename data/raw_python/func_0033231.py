def find_resource_url_basename(self, resource_url):
        '''
        Figure out path basename for given resource_url
        '''
        scheme = resource_url.parsed.scheme
        if scheme in ('http', 'https', 'file'):
            return _get_basename_based_on_url(resource_url)

        elif scheme in ('git', 'git+https', 'git+http'):
            if len(resource_url.args) == 2:
                # For now, git has 2 positional args, hash and path
                git_tree, subpath = resource_url.args
                basename = os.path.basename(subpath)
                if basename:
                    return basename  # subpath was not '/' or ''
        return _get_basename_based_on_url(resource_url)