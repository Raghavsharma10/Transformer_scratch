def get(self, filter_lambda=None, map_lambda=None):
        """Select elements of the TSV, using python filter and map.

        Parameters
        ----------
        filter_lambda : function
            function to filter the tsv rows (the function needs to return True/False)
        map_lambda : function
            function to select the tsv columns

        Returns
        -------
        list
            list (not a generator, because that's the most common case)

        Examples
        --------
        To select all the channels in one list, called "good_labels"::

            >>> file_Tsv.get(lambda x: x['name'] in good_labels)

        To select all the names of the channels:

            >>> file_Tsv.get(map_filter=lambda x: x['name'])

        """
        if filter_lambda is None:
            filter_lambda = lambda x: True
        if map_lambda is None:
            map_lambda = lambda x: x
        return list(map(map_lambda, filter(filter_lambda, self.tsv)))