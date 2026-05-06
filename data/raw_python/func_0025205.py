def create_population(self):
        "Create the initial population"
        base = self._base
        if base._share_inputs:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = used_inputs_var
        if base._pr_variable == 0:
            used_inputs_var = SelectNumbers([])
            used_inputs_naive = SelectNumbers([x for x in range(base.nvar)])
        elif base._pr_variable == 1:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = SelectNumbers([])
        else:
            used_inputs_var = SelectNumbers([x for x in range(base.nvar)])
            used_inputs_naive = SelectNumbers([x for x in range(base.nvar)])
        nb_input = Inputs(base, used_inputs_naive, functions=base._input_functions)
        while ((base._all_inputs and not base.stopping_criteria_tl()) or
               (self.popsize < base.popsize and
                not base.stopping_criteria())):
            if base._all_inputs and used_inputs_var.empty() and used_inputs_naive.empty():
                base._init_popsize = self.popsize
                break
            if nb_input.use_all_variables():
                v = nb_input.all_variables()
                if v is None:
                    continue
            elif not used_inputs_var.empty() and np.random.random() < base._pr_variable:
                v = self.variable_input(used_inputs_var)
                if v is None:
                    used_inputs_var.pos = used_inputs_var.size
                    continue
            elif not used_inputs_naive.empty():
                v = nb_input.input()
                if not used_inputs_var.empty() and used_inputs_naive.empty():
                    base._pr_variable = 1
                if v is None:
                    used_inputs_naive.pos = used_inputs_naive.size
                    if not used_inputs_var.empty():
                        base._pr_variable = 1
                    continue
            else:
                gen = self.generation
                self.generation = 0
                v = base.random_offspring()
                self.generation = gen
            self.add(v)