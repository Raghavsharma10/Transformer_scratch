def _modify_tag(self, name, description, servers, new_name):
        """
        PUT /tag/name. Returns a dict that can be used to create a Tag object.

        Private method used by the Tag class and TagManager.modify_tag.
        """
        body = {'tag': Tag(new_name, description, servers).to_dict()}
        res = self.request('PUT', '/tag/' + name, body)
        return res['tag']