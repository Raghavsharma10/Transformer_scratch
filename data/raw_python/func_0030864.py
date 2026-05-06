def write_sparse(self, file_path: str):
        """Write a sparse representation to a tab-delimited text file.
        
        TODO: docstring"""

        coo = sparse.coo_matrix(self.X)
        data = OrderedDict([(0, coo.row+1), (1, coo.col+1), (2, coo.data)])
        df = pd.DataFrame(data, columns=data.keys())
        with open(file_path, 'w') as ofh:
            ofh.write('%%MatrixMarket matrix coordinate real general\n')
            ofh.write('%%%s\n' % '\t'.join(self.genes.astype(str)))
            ofh.write('%%%s\n' % '\t'.join(self.cells.astype(str)))
            ofh.write('%\n')
            ofh.write('%d %d %d\n' % (coo.shape[0], coo.shape[1], coo.nnz))
            df.to_csv(ofh, sep=' ', float_format='%.5f',
                      header=None, index=None)