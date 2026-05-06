def get_nodes_lines(self, **kwargs):
        """Obtain stop IDs, coordinates and line information.

        Args:
            nodes (list[int] | int): nodes to query, may be empty to get
                all nodes.

        Returns:
            Status boolean and parsed response (list[NodeLinesItem]), or message
            string in case of error.
        """
        # Endpoint parameters
        params = {'Nodes': util.ints_to_string(kwargs.get('nodes', []))}

        # Request
        result = self.make_request('bus', 'get_nodes_lines', **params)

        if not util.check_result(result):
            return False, result.get('resultDescription', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'resultValues')
        return True, [emtype.NodeLinesItem(**a) for a in values]