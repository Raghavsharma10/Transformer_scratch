def set_data(self, adjacency_mat=None, **kwargs):
        """Set the data

        Parameters
        ----------
        adjacency_mat : ndarray | None
            The adjacency matrix.
        **kwargs : dict
            Keyword arguments to pass to the arrows.
        """
        if adjacency_mat is not None:
            if adjacency_mat.shape[0] != adjacency_mat.shape[1]:
                raise ValueError("Adjacency matrix should be square.")

            self._adjacency_mat = adjacency_mat

        for k in self._arrow_attributes:
            if k in kwargs:
                translated = (self._arrow_kw_trans[k] if k in
                              self._arrow_kw_trans else k)

                setattr(self._edges, translated, kwargs.pop(k))

        arrow_kwargs = {}
        for k in self._arrow_kwargs:
            if k in kwargs:
                translated = (self._arrow_kw_trans[k] if k in
                              self._arrow_kw_trans else k)

                arrow_kwargs[translated] = kwargs.pop(k)

        node_kwargs = {}
        for k in self._node_kwargs:
            if k in kwargs:
                translated = (self._node_kw_trans[k] if k in
                              self._node_kw_trans else k)

                node_kwargs[translated] = kwargs.pop(k)

        if len(kwargs) > 0:
            raise TypeError("%s.set_data() got invalid keyword arguments: %S"
                            % (self.__class__.__name__, list(kwargs.keys())))

        # The actual data is set in GraphVisual.animate_layout or
        # GraphVisual.set_final_layout
        self._arrow_data = arrow_kwargs
        self._node_data = node_kwargs

        if not self._animate:
            self.set_final_layout()