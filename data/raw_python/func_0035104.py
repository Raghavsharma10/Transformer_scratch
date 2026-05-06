def set_all_file_column_labels(self, xlabel=None, ylabel=None):
        """Indicate general x,y column labels.

        This sets the general x and y column labels into data files for all plots.
        It can be overridden for specific plots.

        Args:
            xlabel/ylabel (str, optional): String indicating column label for x,y values
                into the data files. Default is None.

        Raises:
            UserWarning: If xlabel and ylabel are both not specified,
                The user will be alerted, but the code will not stop.

        """
        if xlabel is not None:
            self.general.x_column_label = xlabel
        if ylabel is not None:
            self.general.y_column_label = ylabel
        if xlabel is None and ylabel is None:
            warnings.warn("is not specifying x or y lables even"
                          + "though column labels function is called.", UserWarning)
        return