def _single_feature_logliks_one_step(self, feature, models):
        """Get log-likelihood of models at each parameterization for given data

        Parameters
        ----------
        feature : pandas.Series
            Percent-based values of a single feature. May contain NAs, but only
            non-NA values are used.

        Returns
        -------
        logliks : pandas.DataFrame

        """
        x_non_na = feature[~feature.isnull()]
        if x_non_na.empty:
            return pd.DataFrame()
        else:
            dfs = []
            for name, model in models.items():
                df = model.single_feature_logliks(feature)
                df['Modality'] = name
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)