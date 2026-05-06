def update_list(self, list_id, revision, title=None, public=None):
        ''' Updates the list with the given ID to have the given title and public flag '''
        return lists_endpoint.update_list(self, list_id, revision, title=title, public=public)