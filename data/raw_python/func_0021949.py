def return_obj(cols, df, return_cols=False):
        """Construct a DataFrameHolder and then return either that or the DataFrame."""
        df_holder = DataFrameHolder(cols=cols, df=df)
        return df_holder.return_self(return_cols=return_cols)