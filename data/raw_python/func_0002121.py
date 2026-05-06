def _get_item_class(self, url):
        """ Return the model class matching a URL """
        if '/layers/' in url:
            return Layer
        elif '/tables/' in url:
            return Table
        elif '/sets/' in url:
            return Set
        # elif '/documents/' in url:
        #     return Document
        else:
            raise NotImplementedError("No support for catalog results of type %s" % url)