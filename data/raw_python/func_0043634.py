def convert_halo_to_array_form(self, halo):
        """
        Converts the :samp:`{halo}` argument to a :samp:`({self}.array_shape.size, 2)`
        shaped array.

        :type halo: :samp:`None`, :obj:`int`, :samp:`self.array_shape.size` length sequence
            of :samp:`int` or :samp:`(self.array_shape.size, 2)` shaped array
            of :samp:`int`
        :param halo: Halo to be converted to :samp:`(len(self.array_shape), 2)` shaped array form.
        :rtype: :obj:`numpy.ndarray`
        :return: A :samp:`(len(self.array_shape), 2)` shaped array of :obj:`numpy.int64` elements.
        """
        return convert_halo_to_array_form(halo=halo, ndim=len(self.array_shape))