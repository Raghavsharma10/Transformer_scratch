def read_tsv(cls, file_or_buffer: str):
        """Read genes from tab-delimited text file."""
        df = pd.read_csv(file_or_buffer, sep='\t', index_col=0)
        df = df.where(pd.notnull(df), None)
        # Note: df.where(..., None) changes all column types to `object`.
        return cls(df)