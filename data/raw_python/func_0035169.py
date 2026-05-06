def create_plots(self):
        """Creates plots according to each plotting class.

        """
        for i, axis in enumerate(self.ax):
            # plot everything. First check general dict for parameters related to plots.
            trans_plot_class_call = globals()[self.plot_types[i]]
            trans_plot_class = trans_plot_class_call(self.fig, axis,
                                                     self.value_classes[i].x_arr_list,
                                                     self.value_classes[i].y_arr_list,
                                                     self.value_classes[i].z_arr_list,
                                                     colorbar=(
                                                        self.colorbar_classes[self.plot_types[i]]),
                                                     **{**self.general,
                                                        **self.figure,
                                                        **self.plot_info[str(i)],
                                                        **self.plot_info[str(i)]['limits'],
                                                        **self.plot_info[str(i)]['label'],
                                                        **self.plot_info[str(i)]['extra'],
                                                        **self.plot_info[str(i)]['legend']})

            # create the plot
            trans_plot_class.make_plot()

            # setup the plot
            trans_plot_class.setup_plot()

            # print("Axis", i, "Complete")
        return