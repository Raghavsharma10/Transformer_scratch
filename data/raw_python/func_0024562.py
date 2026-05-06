def cfd(self, cycle_data,size_history= None, pointscolumn= None, stacked = True ):
        """Return the data to build a cumulative flow diagram: a DataFrame,
        indexed by day, with columns containing cumulative counts for each
        of the items in the configured cycle.

        In addition, a column called `cycle_time` contains the approximate
        average cycle time of that day based on the first "accepted" status
        and the first "complete" status.

        If stacked = True then return dataframe suitable for plotting as stacked area chart
        else return for platting as non-staked or line chart.
        """

        # Define helper function
        def cumulativeColumnStates(df,stacked):
            """
            Calculate the column sums, were the incoming matrix columns represents items in workflow states
            States progress from left to right.
            We what to zero out items, other than right most value to avoid counting items in prior states.
            :param df:
            :return: pandas dataframe row with sum of column items
            """

            # Helper functions to return the right most cells in 2D array
            def last_number(lst):
                if all(map(lambda x: x == 0, lst)):
                    return 0
                elif lst[-1] != 0:
                    return len(lst) - 1
                else:
                    return last_number(lst[:-1])

            def fill_others(lst):
                new_lst = [0] * len(lst)
                new_lst[last_number(lst)] = lst[last_number(lst)]
                return new_lst

            df_zeroed = df.fillna(value=0)  # ,inplace = True   Get rid of non numeric items. Make a ?deep? copy
            if stacked:
                df_result = df_zeroed.apply(lambda x: fill_others(x.values.tolist()), axis=1)
            else:
                df_result = df_zeroed

            sum_row = df_result[df.columns].sum()  # Sum Columns
            return pd.DataFrame(data=sum_row).T  # Transpose into row dataframe and return

        # Helper function to return the right most cells in 2D array
        def keeprightmoststate(df):
            """
            Incoming matrix columns represents items in workflow states
            States progress from left to right.
            We what to zero out items, other than right most value.
            :param df:
            :return: pandas dataframe row with sum of column items
            """

            def last_number(lst):
                if all(map(lambda x: x == 0, lst)):
                    return 0
                elif lst[-1] != 0:
                    return len(lst) - 1
                else:
                    return last_number(lst[:-1])

            def fill_others(lst):
                new_lst = [0] * len(lst)
                new_lst[last_number(lst)] = lst[last_number(lst)]
                return new_lst

            df_zeroed = df.fillna(value=0)  # ,inplace = True   Get rid of non numeric items. Make a ?deep? copy
            df_result = df_zeroed.apply(lambda x: fill_others(x.values.tolist()), axis=1)
            return df_result

        # Define helper function
        def hide_greater_than_date(cell, adate):
            """ Helper function to compare date values in cells
            """
            result = False
            try:
                celldatetime = datetime.date(cell.year, cell.month, cell.day)
            except:
                return True
            if celldatetime > adate:
                return True
            return False  # We have a date value in cell and it is less than or equal to input date

        # Helper function
        def appendDFToCSV(df, csvFilePath, sep="\t",date_format='%Y-%m-%d', encoding='utf-8'):
            import os
            if not os.path.isfile(csvFilePath):
                df.to_csv(csvFilePath, mode='a', index=False, sep=sep, date_format=date_format, encoding=encoding)
            elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
                raise Exception(
                    "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                        len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
            elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
                raise Exception("Columns and column order of dataframe and csv file do not match!!")
            else:
                df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False, date_format=date_format, encoding=encoding)


        #print(pointscolumn)

        # List of all state change columns that may have date value in them
        cycle_names = [s['name'] for s in self.settings['cycle']]

        # Create list of columns that we want to return in our results dataFrame
        slice_columns = list(self.settings['none_sized_statuses']) # Make a COPY of the list so that we dont modify the reference.
        if pointscolumn:
            for size_state in self.settings['sized_statuses']:  # states_to_size:
                sizedStateName = size_state + 'Sized'
                slice_columns.append(sizedStateName)
            # Check that it works if we use all columns as sized.
            slice_columns = []
            for size_state in cycle_names:
                sizedStateName = size_state + 'Sized'
                slice_columns.append(sizedStateName)
        else:
            slice_columns = cycle_names


        # Build a dataframe of just the "date" columns
        df = cycle_data[cycle_names].copy()

        # Strip out times from all dates
        df = pd.DataFrame(
            np.array(df.values, dtype='<M8[ns]').astype('<M8[D]').astype('<M8[ns]'),
            columns=df.columns,
            index=df.index
        )

        # No history provided this thus we return dataframe with just column headers.
        if size_history is None:
            return df

        # Get a list of dates that a issue changed state
        state_changes_on_dates_set = set()
        for state in cycle_names:
            state_changes_on_dates_set = state_changes_on_dates_set.union(set(df[state]))
            # How many unique days did a issue stage state
        # Remove non timestamp vlaues and sort the list
        state_changes_on_dates = filter(lambda x: type(x.date()) == datetime.date,
                                        sorted(list(state_changes_on_dates_set)))



        # Replace missing NaT values (happens if a status is skipped) with the subsequent timestamp
        df = df.fillna(method='bfill', axis=1)


        if pointscolumn:
            storypoints = cycle_data[pointscolumn] # As at today
            ids = cycle_data['key']


        # create blank results dataframe
        df_results = pd.DataFrame()
        # For each date on which we had a issue state change we want to count and sum the totals for each of the given states
        # 'Open','Analysis','Backlog','In Process','Done','Withdrawn'
        timenowstr = datetime.datetime.now().strftime('-run-%Y-%m-%d_%H-%M-%S')
        for date_index,statechangedate in enumerate(state_changes_on_dates):
            if date_index%10 == 0: # Print out Progress every tenth
                pass #print("CFD state change {} of {} ".format(date_index,len(state_changes_on_dates)))
            if type(statechangedate.date()) == datetime.date:
                # filterdate.year,filterdate.month,filterdate.day
                filterdate = datetime.date(statechangedate.year, statechangedate.month,
                                           statechangedate.day)  # statechangedate.datetime()

                # Apply function to each cell and only make it visible if issue was in state on or after the filter date
                df_filtered = df.applymap(lambda x: 0 if hide_greater_than_date(x, filterdate) else 1)

                if stacked:
                    df_filtered=keeprightmoststate(df_filtered)

                if pointscolumn and (size_history is not None):

                    # For debug
                    #if filterdate.isoformat() == '2016-11-22':
                    #    size_history.loc[filterdate.isoformat()].to_csv("debug-size-history.csv")
                    storypoints_series_on = size_history.loc[filterdate.isoformat()].T
                    df_size_on_day = pd.Series.to_frame(storypoints_series_on)
                    df_size_on_day.columns = [pointscolumn]

                    # Make sure get size data in the same sequence as ids.
                    left = pd.Series.to_frame(ids)
                    right = df_size_on_day
                    result = left.join(right, on=['key'])  # http://pandas.pydata.org/pandas-docs/stable/merging.html\
                    df_countable = pd.concat([result, df_filtered], axis=1)
                    # for debuging and analytics append the days state to file
                    df_countable['date'] = filterdate.isoformat()
                    if stacked:
                        file_name = "daily-cfd-stacked-run-at"+ timenowstr + ".csv"
                    else:
                        file_name = "daily-cfd-run-at" + timenowstr + ".csv"
                    appendDFToCSV(df_countable, file_name )
                else:
                    df_countable = df_filtered

                # Because we size issues with Story Points we need to add some additional columns
                # for each state based on size not just count
                if pointscolumn:
                    for size_state in self.settings['sized_statuses']: #states_to_size:
                        sizedStateName = size_state + 'Sized'
                        df_countable[sizedStateName] = df_countable.apply( lambda row: (row[pointscolumn] * row[size_state] ), axis=1)

                # For debugging write dataframe to sheet for current day.
                #file_name="countable-cfd-for-day-"+ filterdate.isoformat()+timenowstr+".csv"
                #df_countable.to_csv(file_name, sep='\t', encoding='utf-8', quoting=csv.QUOTE_ALL)

                df_slice = df_countable.loc[:,slice_columns].copy()
                df_sub_sum = cumulativeColumnStates(df_slice,stacked)
                final_table = df_sub_sum.rename(index={0: filterdate})

                # append to results
                df_results = df_results.append(final_table)
        df_results.sort_index(inplace=True)

        df= df_results
        # Count number of times each date occurs, preserving column order
        #df = pd.concat({col: df[col].value_counts() for col in df}, axis=1)[cycle_names]

        # Fill missing dates with 0 and run a cumulative sum
        #df = df.fillna(0).cumsum(axis=0)

        # Reindex to make sure we have all dates
        start, end = df.index.min(), df.index.max()
        try: # If we have no change history we will not have any data in the df and will get a ValueError on reindex
            df = df.reindex(pd.date_range(start, end, freq='D'), method='ffill')
        except ValueError:
            pass

        return df