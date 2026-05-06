def assign_node_labels_and_sizes(self):
        "assign features of nodes to be plotted based on user kwargs"

        # shorthand
        nvals = self.ttree.get_node_values()

        # False == Hide nodes and labels unless user entered size 
        if self.style.node_labels is False:
            self.node_labels = ["" for i in nvals]           
            if self.style.node_sizes is not None:
                if isinstance(self.style.node_sizes, (list, tuple, np.ndarray)):
                    assert len(self.node_sizes) == len(self.style.node_sizes)
                    self.node_sizes = self.style.node_sizes

                elif isinstance(self.style.node_sizes, (int, str)):
                    self.node_sizes = (
                        [int(self.style.node_sizes)] * len(nvals)
                    )
                self.node_labels = [" " if i else "" for i in self.node_sizes]
                    
                    
        # True == Show nodes, label=idx, and show hover
        elif self.style.node_labels is True:
            # turn on node hover even if user did not set it explicit
            self.style.node_hover = True

            # get idx labels
            self.node_labels = self.ttree.get_node_values('idx', 1, 1)

            # use default node size as a list if not provided
            if not self.style.node_sizes:
                self.node_sizes = [18] * len(nvals)
            else:
                assert isinstance(self.style.node_sizes, (int, str))
                self.node_sizes = (
                    [int(self.style.node_sizes)] * len(nvals)
                )

        # User entered lists or other for node labels or sizes; check lengths.
        else:
            # make node labels into a list of values 
            if isinstance(self.style.node_labels, list):
                assert len(self.style.node_labels) == len(nvals)
                self.node_labels = self.style.node_labels

            # check if user entered a feature else use entered val
            elif isinstance(self.style.node_labels, str):
                self.node_labels = [self.style.node_labels] * len(nvals)
                if self.style.node_labels in self.ttree.features:
                    self.node_labels = self.ttree.get_node_values(
                        self.style.node_labels, 1, 0)

            # default to idx at internals if nothing else
            else:
                self.node_labels = self.ttree.get_node_values("idx", 1, 0)

            # make node sizes as a list; set to zero if node label is ""
            if isinstance(self.style.node_sizes, list):
                assert len(self.style.node_sizes) == len(nvals)
                self.node_sizes = self.style.node_sizes
            elif isinstance(self.style.node_sizes, (str, int, float)):
                self.node_sizes = [int(self.style.node_sizes)] * len(nvals)
            else:
                self.node_sizes = [18] * len(nvals)

            # override node sizes to hide based on node labels
            for nidx, node in enumerate(self.node_labels):
                if self.node_labels[nidx] == "":
                    self.node_sizes[nidx] = 0

        # ensure string type
        self.node_labels = [str(i) for i in self.node_labels]