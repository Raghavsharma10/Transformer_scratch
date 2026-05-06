def clear_global(self):
        """Clear only any cached global data.

        """
        vname = self.varname
        logger.debug(f'global clearning {vname}')
        if vname in globals():
            logger.debug('removing global instance var: {}'.format(vname))
            del globals()[vname]