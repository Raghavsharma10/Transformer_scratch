def add_dataset(self, name=None, label=None,
                    x_column_label=None, y_column_label=None, index=None, control=False):
        """Add a dataset to a specific plot.

        This method adds a dataset to a plot. Its functional use is imperative
        to the plot generation. It handles adding new files as well
        as indexing to files that are added to other plots.

        All Args default to None. However, these are note the defaults
        in the code. See DataImportContainer attributes for defaults in code.

        Args:
            name (str, optional): Name (path) for file.
                Required if reading from a file (at least one).
                Required if file_name is not in "general". Must be ".txt" or ".hdf5".
                Can include path from working directory.
            label (str, optional): Column label in the dataset corresponding to desired SNR value.
                Required if reading from a file (at least one).
            x_column_label/y_column_label (str, optional): Column label from input file identifying
                x/y values. This can override setting in "general". Default
                is `x`/`y`.
            index (int, optional): Index of plot with preloaded data.
                Required if not loading a file.
            control (bool, optional): If True, this dataset is set to the control.
                This is needed for Ratio plots. It sets
                the baseline. Default is False.

        Raises:
            ValueError: If no options are passes. This means no file indication
                nor index.

        """
        if name is None and label is None and index is None:
            raise ValueError("Attempting to add a dataset without"
                             + "supplying index or file information.")

        if index is None:
            trans_dict = DataImportContainer()
            if name is not None:
                trans_dict.file_name = name

            if label is not None:
                trans_dict.label = label

            if x_column_label is not None:
                trans_dict.x_column_label = x_column_label

            if y_column_label is not None:
                trans_dict.y_column_label = y_column_label

            if control:
                self.control = trans_dict
            else:
                # need to append file to file list.
                if 'file' not in self.__dict__:
                    self.file = []
                self.file.append(trans_dict)
        else:
            if control:
                self.control = DataImportContainer()
                self.control.index = index

            else:
                # need to append index to index list.
                if 'indices' not in self.__dict__:
                    self.indices = []

                self.indices.append(index)
        return