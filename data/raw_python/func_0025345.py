def fitness_vs(self, v):
        """Fitness function in the validation set
        In classification it uses BER and RSE in regression"""
        base = self._base
        if base._classifier:
            if base._multiple_outputs:
                v.fitness_vs = v._error
                # if base._fitness_function == 'macro-F1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'BER':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'macro-Precision':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'accDotMacroF1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'macro-RecallF1':
                #     v.fitness_vs = v._error
                # elif base._fitness_function == 'F1':
                #     v.fitness_vs = v._error
                # else:
                #     v.fitness_vs = - v._error.dot(base._mask_vs) / base._mask_vs.sum()
            else:
                v.fitness_vs = -((base.y - v.hy.sign()).sign().fabs() *
                                 base._mask_vs).sum()
        else:
            mask = base._mask
            y = base.y
            hy = v.hy
            if not isinstance(mask, list):
                mask = [mask]
                y = [y]
                hy = [hy]
            fit = []
            for _mask, _y, _hy in zip(mask, y, hy):
                m = (_mask + -1).fabs()
                x = _y * m
                y = _hy * m
                a = (x - y).sq().sum()
                b = (x + -x.sum() / x.size()).sq().sum()
                fit.append(-a / b)
            v.fitness_vs = np.mean(fit)