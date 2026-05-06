def ref_frequency(self, context):
        """ Return a reference frequency array to montblanc """
        ref_freq = np.empty(context.shape, context.dtype)
        ref_freq[:] = 1.415e9

        return ref_freq