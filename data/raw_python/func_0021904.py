def load_dataframe(self, df_loader_name):
        """
        Instead of joining a DataFrameJoiner with the Cohort in `as_dataframe`, sometimes
        we may want to just directly load a particular DataFrame.
        """
        logger.debug("loading dataframe: {}".format(df_loader_name))
        # Get the DataFrameLoader object corresponding to this name.
        df_loaders = [df_loader for df_loader in self.df_loaders if df_loader.name == df_loader_name]

        if len(df_loaders) == 0:
            raise ValueError("No DataFrameLoader with name %s" % df_loader_name)
        if len(df_loaders) > 1:
            raise ValueError("Multiple DataFrameLoaders with name %s" % df_loader_name)

        return df_loaders[0].load_dataframe()