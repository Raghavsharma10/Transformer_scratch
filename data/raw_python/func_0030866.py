def read_10xgenomics(cls, tarball_fpath: str, prefix: str,
                         use_ensembl_ids: bool = False):
        """Read a 10X genomics compressed tarball containing expression data.
        
        Note: common prefix patterns:
        - "filtered_gene_bc_matrices/[annotations]/"
        - "filtered_matrices_mex/[annotations]/"

        TODO: docstring"""

        _LOGGER.info('Reading file: %s', tarball_fpath)

        with tarfile.open(tarball_fpath, mode='r:gz') as tf:
            ti = tf.getmember('%smatrix.mtx' % prefix)
            with tf.extractfile(ti) as fh:
                mtx = scipy.io.mmread(fh)

            ti = tf.getmember('%sgenes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                i = 1
                if use_ensembl_ids:
                    i = 0
                gene_names = \
                        [row[i] for row in csv.reader(wrapper, delimiter='\t')]

            ti = tf.getmember('%sbarcodes.tsv' % prefix)
            with tf.extractfile(ti) as fh:
                wrapper = io.TextIOWrapper(fh, encoding='ascii')
                barcodes = \
                        [row[0] for row in csv.reader(wrapper, delimiter='\t')]


            assert mtx.shape[0] == len(gene_names)
            assert mtx.shape[1] == len(barcodes)
        
        _LOGGER.info('Matrix dimensions: %s', str(mtx.shape))
        X = mtx.todense()
        matrix = cls(X=X, genes=gene_names, cells=barcodes)
        
        return matrix