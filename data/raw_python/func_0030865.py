def read_sparse(cls, file_path: str):
        """Read a sparse representation from a tab-delimited text file.
        
        TODO: docstring"""
        
        with open(file_path) as fh:
            next(fh)  # skip header line
            genes = next(fh)[1:-1].split('\t')
            cells = next(fh)[1:-1].split('\t')
            next(fh)
            m, n, nnz = [int(s) for s in next(fh)[:-1].split(' ')]
        
        t = pd.read_csv(file_path, sep=' ', skiprows=5, header=None,
                        dtype={0: np.uint32, 1: np.uint32})
        
        i = t[0].values - 1
        j = t[1].values - 1
        data = t[2].values

        assert data.size == nnz

        X = sparse.coo_matrix((data, (i,j)), shape=[m, n]).todense()

        return cls(X=X, genes=genes, cells=cells)