def difference(self, instrument1, instrument2, bounds, data_labels,
                   cost_function):
        """
        Calculates the difference in signals from multiple
        instruments within the given bounds.

        Parameters
        ----------
        instrument1 : Instrument
            Information must already be loaded into the
            instrument.

        instrument2 : Instrument
            Information must already be loaded into the
            instrument.

        bounds : list of tuples in the form (inst1_label, inst2_label,
            min, max, max_difference)
            inst1_label are inst2_label are labels for the data in
            instrument1 and instrument2
            min and max are bounds on the data considered
            max_difference is the maximum difference between two points
            for the difference to be calculated

        data_labels : list of tuples of data labels
            The first key is used to access data in s1
            and the second data in s2.

        cost_function : function
            function that operates on two rows of the instrument data.
            used to determine the distance between two points for finding
            closest points

        Returns
        -------
        data_df: pandas DataFrame
            Each row has a point from instrument1, with the keys
            preceded by '1_', and a point within bounds on that point
            from instrument2 with the keys preceded by '2_', and the 
            difference between the instruments' data for all the labels
            in data_labels

        Created as part of a Spring 2018 UTDesign project.
        """
        
        """
        Draft Pseudocode
        ----------------
        Check integrity of inputs.

        Let STD_LABELS be the constant tuple:
        ("time", "lat", "long", "alt")

        Note: modify so that user can override labels for time,
        lat, long, data for each satelite.

        // We only care about the data currently loaded
           into each object.

        Let start be the later of the datetime of the
         first piece of data loaded into s1, the first
         piece of data loaded into s2, and the user
         supplied start bound.

        Let end be the later of the datetime of the first
         piece of data loaded into s1, the first piece
         of data loaded into s2, and the user supplied
         end bound.

        If start is after end, raise an error.

        // Let data be the 2D array of deques holding each piece
        //  of data, sorted into bins by lat/long/alt.

        Let s1_data (resp s2_data) be data from s1.data, s2.data
        filtered by user-provided lat/long/alt bounds, time bounds
        calculated.

        Let data be a dictionary of lists with the keys
        [ dl1 for dl1, dl2 in data_labels ] +
        STD_LABELS +
        [ lb+"2" for lb in STD_LABELS ]

        For each piece of data s1_point in s1_data:

            # Hopefully np.where is very good, because this
            #  runs O(n) times.
            # We could try reusing selections, maybe, if needed.
            #  This would probably involve binning.
            Let s2_near be the data from s2.data within certain
             bounds on lat/long/alt/time using 8 statements to
             numpy.where. We can probably get those defaults from
             the user or handy constants / config?

            # We could try a different algorithm for closest pairs
            # of points.

            Let distance be the numpy array representing the
             distance between s1_point and each point in s2_near.

            # S: Difference for others: change this line.
            For each of those, calculate the spatial difference
             from the s1 using lat/long/alt. If s2_near is
             empty; break loop.

            Let s2_nearest be the point in s2_near corresponding
             to the lowest distance.

            Append to data: a point, indexed by the time from
             s1_point, containing the following data:

            # note
            Let n be the length of data["time"].
            For each key in data:
                Assert len(data[key]) == n
            End for.

            # Create data row to pass to pandas.
            Let row be an empty dict.
            For dl1, dl2 in data_labels:
                Append s1_point[dl1] - s2_nearest[dl2] to data[dl1].

            For key in STD_LABELS:
                Append s1_point[translate[key]] to data[key]
                key = key+"2"
                Append s2_nearest[translate[key]] to data[key]

        Let data_df be a pandas dataframe created from the data
        in data.

        return { 'data': data_df, 'start':start, 'end':end }
        """

        labels = [dl1 for dl1, dl2 in data_labels] + ['1_'+b[0] for b in bounds] + ['2_'+b[1] for b in bounds] + ['dist']
        data = {label: [] for label in labels}

        # Apply bounds
        inst1 = instrument1.data
        inst2 = instrument2.data
        for b in bounds:
            label1 = b[0]
            label2 = b[1]
            low = b[2]
            high = b[3]

            data1 = inst1[label1]
            ind1 = np.where((data1 >= low) & (data1 < high))
            inst1 = inst1.iloc[ind1]

            data2 = inst2[label2]
            ind2 = np.where((data2 >= low) & (data2 < high))
            inst2 = inst2.iloc[ind2]

        for i, s1_point in inst1.iterrows():
            # Gets points in instrument2 within the given bounds
            s2_near = instrument2.data
            for b in bounds:
                label1 = b[0]
                label2 = b[1]
                s1_val = s1_point[label1]
                max_dist = b[4]
                minbound = s1_val - max_dist
                maxbound = s1_val + max_dist

                data2 = s2_near[label2]
                indices = np.where((data2 >= minbound) & (data2 < maxbound))
                s2_near = s2_near.iloc[indices]

            # Finds nearest point to s1_point in s2_near
            s2_nearest = None
            min_dist = float('NaN')
            for j, s2_point in s2_near.iterrows():
                dist = cost_function(s1_point, s2_point)
                if dist < min_dist or min_dist != min_dist:
                    min_dist = dist
                    s2_nearest = s2_point

            data['dist'].append(min_dist)

            # Append difference to data dict
            for dl1, dl2 in data_labels:
                if s2_nearest is not None:
                    data[dl1].append(s1_point[dl1] - s2_nearest[dl2])
                else:
                    data[dl1].append(float('NaN'))

            # Append the rest of the row
            for b in bounds:
                label1 = b[0]
                label2 = b[1]
                data['1_'+label1].append(s1_point[label1])
                if s2_nearest is not None:
                    data['2_'+label2].append(s2_nearest[label2])
                else:
                    data['2_'+label2].append(float('NaN'))

        data_df = pds.DataFrame(data=data)
        return data_df