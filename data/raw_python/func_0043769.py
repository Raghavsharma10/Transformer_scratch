def _reconstruct(self, path_to_root):

        '''
            a helper method for finding the schema endpoint from a path to root

        :param path_to_root: string with dot path to root from
        :return: list, dict, string, number, or boolean at path to root
        '''

    # split path to root into segments
        item_pattern = re.compile('\d+\\]')
        dot_pattern = re.compile('\\.|\\[')
        path_segments = dot_pattern.split(path_to_root)

    # construct base schema endpoint
        schema_endpoint = self.schema

    # reconstruct schema endpoint from segments
        if path_segments[1]:
            for i in range(1,len(path_segments)):
                if item_pattern.match(path_segments[i]):
                    schema_endpoint = schema_endpoint[0]
                else:
                    schema_endpoint = schema_endpoint[path_segments[i]]

        return schema_endpoint