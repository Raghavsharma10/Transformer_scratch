def save(func):
    """@decorator: Saves data after executing :func:.

    Also performs modifications set as permanent options.

    """
    def aux(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        path = (hasattr(self, 'path') and self.path
                or os.path.join(os.getcwd(), '.td'))
        gpath = (hasattr(self, 'gpath') and self.gpath
                 or os.path.expanduser('~/.tdrc'))
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(os.path.dirname(path), '.td~'))
        open(path, 'w').write(
            json.dumps({
                'items': self.data,
                'refs': self.refs,
                'options': self.options
            })
        )
        open(gpath, 'w').write(json.dumps(self.globalOptions))
        return out
    return aux