def get_doc(self, tag_name):
        "Get documentation for the first tag matching the given name"
        for tag,func in self.tags:
            if tag.startswith(tag_name) and func.__doc__:
                return func.__doc__