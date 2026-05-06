def fitness(self, v):
        "Fitness function in the training set"
        base = self._base
        if base._classifier:
            if base._multiple_outputs:
                hy = SparseArray.argmax(v.hy)
                fit_func = base._fitness_function
                if fit_func == 'macro-F1' or fit_func == 'a_F1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_F1(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotF1' or fit_func == 'g_F1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_F1(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotRecallDotPrecision' or fit_func == 'g_g_recall_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_g_recall_precision(base._y_klass, hy,
                                                               base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'BER' or fit_func == 'a_recall':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_recall(base._y_klass, hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotRecall' or fit_func == 'g_recall':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_recall(base._y_klass, hy,
                                                   base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'macro-Precision' or fit_func == 'a_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.a_precision(base._y_klass, hy,
                                                      base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'DotPrecision' or fit_func == 'g_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_precision(base._y_klass, hy,
                                                      base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'accDotMacroF1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.accDotMacroF1(base._y_klass, hy,
                                                        base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'macro-RecallF1':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.macroRecallF1(base._y_klass, hy,
                                                        base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'F1':
                    f1_score = self.score
                    f1_index = self._base._F1_index
                    index = self.min_class if f1_index < 0 else f1_index
                    mf1, mf1_v = f1_score.F1(index, base._y_klass,
                                             hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'RecallDotPrecision' or fit_func == 'g_recall_precision':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.g_recall_precision(self.min_class,
                                                             base._y_klass,
                                                             hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                elif fit_func == 'ER' or fit_func == 'accuracy':
                    f1_score = self.score
                    mf1, mf1_v = f1_score.accuracy(base._y_klass,
                                                   hy, base._mask_ts.index)
                    v._error = mf1_v - 1
                    v.fitness = mf1 - 1
                else:
                    raise RuntimeError('Unknown fitness function %s' % base._fitness_function)
            else:
                v.fitness = -base._ytr.SSE(v.hy * base._mask)
        else:
            if base._multiple_outputs:
                _ = np.mean([a.SAE(b.mul(c)) for a, b, c in zip(base._ytr, v.hy, base._mask)])
                v.fitness = - _
            else:
                v.fitness = -base._ytr.SAE(v.hy * base._mask)