def size_history(self,size_data):
        """Return the a DataFrame,
        indexed by day, with columns containing story size for each issue.

        In addition, columns are soted by Jira Issue key. First by Project and then by id number.
        """

        def my_merge(df1, df2):
            # http://stackoverflow.com/questions/34411495/pandas-merge-several-dataframes
            res = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
            cols = sorted(res.columns)
            pairs = []
            for col1, col2 in zip(cols[:-1], cols[1:]):
                if col1.endswith('_x') and col2.endswith('_y'):
                    pairs.append((col1, col2))
            for col1, col2 in pairs:
                res[col1[:-2]] = res[col1].combine_first(res[col2])
                res = res.drop([col1, col2], axis=1)
            return res

        dfs_key = []
        # Group the dataframe by regiment, and for each regiment,
        for name, group in size_data.groupby('key'):
            dfs = []
            for row in group.itertuples():
                # print(row.Index, row.fromDate,row.toDate, row.size)
                dates = pd.date_range(start=row.fromDate, end=row.toDate)
                sizes = [row.size] * len(dates)
                data = {'date': dates, 'size': sizes}
                df2 = pd.DataFrame(data, columns=['date', 'size'])
                pd.to_datetime(df2['date'], format=('%Y-%m-%d'))
                df2.set_index(['date'], inplace=True)
                dfs.append(df2)
            # df_final = reduce(lambda left,right: pd.merge(left,right), dfs)
            df_key = (reduce(my_merge, dfs))
            df_key.columns = [name if x == 'size' else x for x in df_key.columns]
            dfs_key.append(df_key)
        df_all = (reduce(my_merge, dfs_key))

        # Sort the columns based on Jira Project code and issue number
        mykeys = df_all.columns.values.tolist()
        mykeys.sort(key=lambda x: x.split('-')[0] + '-' + str(int(x.split('-')[1])).zfill(6))
        df_all = df_all[mykeys]

        # Reindex to make sure we have all dates
        start, end = df_all.index.min(), df_all.index.max()
        df_all = df_all.reindex(pd.date_range(start, end, freq='D'), method='ffill')

        return df_all