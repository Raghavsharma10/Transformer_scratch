def rms(self, x, params=()):
        """ Returns root mean square value of f(x, params) """
        internal_x, internal_params = self.pre_process(np.asarray(x),
                                                       np.asarray(params))
        if internal_params.ndim > 1:
            raise NotImplementedError("Parameters should be constant.")
        result = np.empty(internal_x.size//self.nx)
        for idx in range(internal_x.shape[0]):
            result[idx] = np.sqrt(np.mean(np.square(self.f_cb(
                internal_x[idx, :], internal_params))))
        return result