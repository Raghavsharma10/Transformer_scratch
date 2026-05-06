def run(self, *args, **kw):
        """ This may be overridden by a subclass. """
        if self._runFunc is not None:
            # remove the two args sent by EditParDialog which we do not use
            if 'mode' in kw: kw.pop('mode')
            if '_save' in kw: kw.pop('_save')
            return self._runFunc(self, *args, **kw)
        else:
            raise taskpars.NoExecError('No way to run task "'+self.__taskName+\
                '". You must either override the "run" method in your '+ \
                'ConfigObjPars subclass, or you must supply a "run" '+ \
                'function in your package.')