def build_subtree_strut(self, result, *args, **kwargs):
        """
        Returns a dictionary in form of
        {node:Resource, children:{node_id: Resource}}

        :param result:
        :return:
        """
        return self.service.build_subtree_strut(result=result, *args, **kwargs)