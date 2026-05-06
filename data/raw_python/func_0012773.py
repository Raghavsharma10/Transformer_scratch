def setidd(cls, iddinfo, iddindex, block, idd_version):
        """Set the IDD to be used by eppy.

        Parameters
        ----------
        iddinfo : list
            Comments and metadata about fields in the IDD.
        block : list
            Field names in the IDD.

        """
        cls.idd_info = iddinfo
        cls.block = block
        cls.idd_index = iddindex
        cls.idd_version = idd_version