def include(self, target):
        """Includes target contents into config.

        :param str|unicode|Section|list target: File path or Section to include.
        """
        for target_ in listify(target):
            if isinstance(target_, Section):
                target_ = ':' + target_.name

            self._set('ini', target_, multi=True)

        return self