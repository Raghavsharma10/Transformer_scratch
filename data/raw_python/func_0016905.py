def point_stokes(self, context):
        """ Return a stokes parameter array to montblanc """
        stokes = np.empty(context.shape, context.dtype)
        stokes[:,:,0] = 1
        stokes[:,:,1:4] = 0
        return stokes