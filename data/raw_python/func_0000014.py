def debug_interactive_inspect_node(self):
        """
        Call after segmentation to see selected node neighborhood.
        User have to select one node by click.
        :return:
        """
        if (
            np.sum(
                np.abs(
                    np.asarray(self.msinds.shape) - np.asarray(self.segmentation.shape)
                )
            )
            == 0
        ):
            segmentation = self.segmentation
        else:
            segmentation = self.temp_msgc_resized_segmentation

        logger.info("Click to select one voxel of interest")
        import sed3

        ed = sed3.sed3(self.msinds, contour=segmentation == 0)
        ed.show()
        edseeds = ed.seeds
        node_msindex = get_node_msindex(self.msinds, edseeds)

        node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = self.debug_inspect_node(
            node_msindex
        )
        import sed3

        ed = sed3.sed3(
            self.msinds, contour=segmentation == 0, seeds=node_neighboor_seeds
        )
        ed.show()

        return (
            node_unariesalt,
            node_neighboor_edges_and_weights,
            node_neighboor_seeds,
            node_msindex,
        )