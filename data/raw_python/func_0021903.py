def as_dataframe(self, on=None, join_with=None, join_how=None,
                     return_cols=False, rename_cols=False,
                     keep_paren_contents=True, **kwargs):
        """
        Return this Cohort as a DataFrame, and optionally include additional columns
        using `on`.

        on : str or function or list or dict, optional
            - A column name.
            - Or a function that creates a new column for comparison, e.g. count.snv_count.
            - Or a list of column-generating functions or column names.
            - Or a map of new column names to their column-generating functions or column names.

        If `on` is a function or functions, kwargs is passed to those functions.
        Otherwise kwargs is ignored.

        Other parameters
        ----------------
        `return_cols`: (bool)
            If True, return column names generated via `on` along with the `DataFrame`
            as a `DataFrameHolder` tuple.
        `rename_cols`: (bool)
            If True, then return columns using "stripped" column names
            ("stripped" means lower-case names without punctuation other than `_`)
            See `utils.strip_column_names` for more details
            defaults to False
        `keep_paren_contents`: (bool)
            If True, then contents of column names within parens are kept.
            If False, contents of column names within-parens are dropped.
            Defaults to True
        ----------

        Return : `DataFrame` (or `DataFrameHolder` if `return_cols` is True)
        """
        df = self._as_dataframe_unmodified(join_with=join_with, join_how=join_how)
        if on is None:
            return DataFrameHolder.return_obj(None, df, return_cols)

        if type(on) == str:
            return DataFrameHolder.return_obj(on, df, return_cols)

        def apply_func(on, col, df):
            """
            Sometimes we have functions that, by necessity, have more parameters
            than just `row`. We construct a function with just the `row` parameter
            so it can be sent to `DataFrame.apply`. We hackishly pass `cohort`
            (as `self`) along if the function accepts a `cohort` argument.
            """
            on_argnames = on.__code__.co_varnames
            if "cohort" not in on_argnames:
                func = lambda row: on(row=row, **kwargs)
            else:
                func = lambda row: on(row=row, cohort=self, **kwargs)

            if self.show_progress:
                tqdm.pandas(desc=col)
                df[col] = df.progress_apply(func, axis=1) ## depends on tqdm on prev line
            else:
                df[col] = df.apply(func, axis=1)
            return DataFrameHolder(col, df)

        def func_name(func, num=0):
            return func.__name__ if not is_lambda(func) else "column_%d" % num

        def is_lambda(func):
            return func.__name__ == (lambda: None).__name__

        if type(on) == FunctionType:
            return apply_func(on, func_name(on), df).return_self(return_cols)

        if len(kwargs) > 0:
            logger.warning("Note: kwargs used with multiple functions; passing them to all functions")

        if type(on) == dict:
            cols = []
            for key, value in on.items():
                if type(value) == str:
                    df[key] = df[value]
                    col = key
                elif type(value) == FunctionType:
                    col, df = apply_func(on=value, col=key, df=df)
                else:
                    raise ValueError("A value of `on`, %s, is not a str or function" % str(value))
                cols.append(col)
        if type(on) == list:
            cols = []
            for i, elem in enumerate(on):
                if type(elem) == str:
                    col = elem
                elif type(elem) == FunctionType:
                    col = func_name(elem, i)
                    col, df = apply_func(on=elem, col=col, df=df)
                cols.append(col)

        if rename_cols:
            rename_dict = _strip_column_names(df.columns, keep_paren_contents=keep_paren_contents)
            df.rename(columns=rename_dict, inplace=True)
            cols = [rename_dict[col] for col in cols]
        return DataFrameHolder(cols, df).return_self(return_cols)