def find_resources(self, rsrc_type, sort=None, yield_pages=False, **kwargs):
        """Find instances of `rsrc_type` that match the filter in `**kwargs`"""
        return rsrc_type.find(self, sort=sort, yield_pages=yield_pages, **kwargs)