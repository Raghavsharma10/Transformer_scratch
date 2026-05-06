def input_data(self):
        """Function to extract data from files according to pid.

        This function will read in the data with
        :class:`bowie.plotutils.readdata.ReadInData`.

        """
        ordererd = np.sort(np.asarray(list(self.plot_info.keys())).astype(int))

        trans_cont_dict = OrderedDict()
        for i in ordererd:
            trans_cont_dict[str(i)] = self.plot_info[str(i)]

        self.plot_info = trans_cont_dict

        # set empty lists for x,y,z
        x = [[]for i in np.arange(len(self.plot_info.keys()))]
        y = [[] for i in np.arange(len(self.plot_info.keys()))]
        z = [[] for i in np.arange(len(self.plot_info.keys()))]

        # read in base files/data
        for k, axis_string in enumerate(self.plot_info.keys()):

            if 'file' not in self.plot_info[axis_string].keys():
                continue
            for j, file_dict in enumerate(self.plot_info[axis_string]['file']):
                data_class = ReadInData(**{**self.general, **file_dict,
                                           **self.plot_info[axis_string]['limits']})

                x[k].append(data_class.x_append_value)
                y[k].append(data_class.y_append_value)
                z[k].append(data_class.z_append_value)

            # print(axis_string)

        # add data from plots to current plot based on index
        for k, axis_string in enumerate(self.plot_info.keys()):

            # takes first file from plot
            if 'indices' in self.plot_info[axis_string]:
                if type(self.plot_info[axis_string]['indices']) == int:
                    self.plot_info[axis_string]['indices'] = (
                        [self.plot_info[axis_string]['indices']])

                for index in self.plot_info[axis_string]['indices']:

                    index = int(index)

                    x[k].append(x[index][0])
                    y[k].append(y[index][0])
                    z[k].append(z[index][0])

        # read or append control values for ratio plots
        for k, axis_string in enumerate(self.plot_info.keys()):
            if 'control' in self.plot_info[axis_string]:
                if ('name' in self.plot_info[axis_string]['control'] or
                        'label' in self.plot_info[axis_string]['control']):
                    file_dict = self.plot_info[axis_string]['control']
                    if 'limits' in self.plot_info[axis_string].keys():
                        liimits_dict = self.plot_info[axis_string]['limits']

                    data_class = ReadInData(**{**self.general, **file_dict,
                                               **self.plot_info[axis_string]['limits']})
                    x[k].append(data_class.x_append_value)
                    y[k].append(data_class.y_append_value)
                    z[k].append(data_class.z_append_value)

                elif 'index' in self.plot_info[axis_string]['control']:
                    index = int(self.plot_info[axis_string]['control']['index'])

                    x[k].append(x[index][0])
                    y[k].append(y[index][0])
                    z[k].append(z[index][0])

            # print(axis_string)

        # transfer lists in PlotVals class.
        value_classes = []
        for k in range(len(x)):
            value_classes.append(PlotVals(x[k], y[k], z[k]))

        self.value_classes = value_classes
        return