def to_json(self, filename=None):
        """
        Exports statistical data to a JSON formatted file

        Parameters
        ----------
        filename:    output file that holds statistics data
        """
        def json_encoder(obj):
            if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                if isinstance(obj.index, pd.core.index.MultiIndex):
                    obj = obj.reset_index()  # convert MultiIndex to columns

                return json.loads(obj.to_json(date_format='iso'))
            elif isinstance(obj, melodist.cascade.CascadeStatistics):
                return obj.__dict__
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                raise TypeError('%s not supported' % type(obj))

        d = dict(
            temp=self.temp,
            wind=self.wind,
            precip=self.precip,
            hum=self.hum,
            glob=self.glob
        )

        j = json.dumps(d, default=json_encoder, indent=4)

        if filename is None:
            return j
        else:
            with open(filename, 'w') as f:
                f.write(j)