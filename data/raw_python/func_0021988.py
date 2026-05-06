def _check_regr(self, regr, new_reg):
        """
        Check that a registration response contains the registration we were
        expecting.
        """
        body = getattr(new_reg, 'body', new_reg)
        for k, v in body.items():
            if k == 'resource' or not v:
                continue
            if regr.body[k] != v:
                raise errors.UnexpectedUpdate(regr)
        if regr.body.key != self.key.public_key():
            raise errors.UnexpectedUpdate(regr)
        return regr