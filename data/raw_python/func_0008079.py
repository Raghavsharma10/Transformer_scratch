def set_implementation(self, impl):
        """
        Sets the implementation of this module

        Parameters
        ----------
        impl : str
            One of ["python", "c"]

        """
        if impl.lower() == 'python':
            self.__impl__ = self.__IMPL_PYTHON__
        elif impl.lower() == 'c':
            self.__impl__ = self.__IMPL_C__
        else:
            import warnings
            warnings.warn('Implementation '+impl+' is not known. Using the fallback python implementation.')
            self.__impl__ = self.__IMPL_PYTHON__