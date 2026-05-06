def fast_substitute(monomial, old_sub, new_sub):
    """Experimental fast substitution routine that considers only restricted
    cases of noncommutative algebras. In rare cases, it fails to find a
    substitution. Use it with proper testing.

    :param monomial: The monomial with parts need to be substituted.
    :param old_sub: The part to be replaced.
    :param new_sub: The replacement.
    """
    if is_number_type(monomial):
        return monomial
    if monomial.is_Add:
        return sum([fast_substitute(element, old_sub, new_sub) for element in
                    monomial.as_ordered_terms()])

    comm_factors, ncomm_factors = split_commutative_parts(monomial)
    old_comm_factors, old_ncomm_factors = split_commutative_parts(old_sub)
    # This is a temporary hack
    if not isinstance(new_sub, (int, float, complex)):
        new_comm_factors, _ = split_commutative_parts(new_sub)
    else:
        new_comm_factors = [new_sub]
    comm_monomial = 1
    is_constant_term = False
    if comm_factors != ():
        if len(comm_factors) == 1 and is_number_type(comm_factors[0]):
            is_constant_term = True
            comm_monomial = comm_factors[0]
        else:
            for comm_factor in comm_factors:
                comm_monomial *= comm_factor
            if old_comm_factors != ():
                comm_old_sub = 1
                for comm_factor in old_comm_factors:
                    comm_old_sub *= comm_factor
                comm_new_sub = 1
                for comm_factor in new_comm_factors:
                    comm_new_sub *= comm_factor
                # Dummy heuristic to get around retarded SymPy bug
                if isinstance(comm_old_sub, Pow):
                    # In this case, we are in trouble
                    old_base = comm_old_sub.base
                    if comm_monomial.has(old_base):
                        old_degree = comm_old_sub.exp
                        new_monomial = 1
                        match = False
                        for factor in comm_monomial.as_ordered_factors():
                            if factor.has(old_base):
                                if isinstance(factor, Pow):
                                    degree = factor.exp
                                    if degree >= old_degree:
                                        match = True
                                        new_monomial *= \
                                            old_base**(degree-old_degree) * \
                                            comm_new_sub

                                else:
                                    new_monomial *= factor
                            else:
                                new_monomial *= factor
                        if match:
                            comm_monomial = new_monomial
                else:
                    comm_monomial = comm_monomial.subs(comm_old_sub,
                                                       comm_new_sub)
    if ncomm_factors == () or old_ncomm_factors == ():
        return comm_monomial
    # old_factors = old_sub.as_ordered_factors()
    # factors = monomial.as_ordered_factors()
    new_var_list = []
    new_monomial = 1
    left_remainder = 1
    right_remainder = 1
    for i in range(len(ncomm_factors) - len(old_ncomm_factors) + 1):
        for j, old_ncomm_factor in enumerate(old_ncomm_factors):
            ncomm_factor = ncomm_factors[i + j]
            if isinstance(ncomm_factor, Symbol) and \
                (isinstance(old_ncomm_factor, Operator) or
                 (isinstance(old_ncomm_factor, Symbol) and
                  ncomm_factor != old_ncomm_factor)):
                break
            if isinstance(ncomm_factor, Operator) and \
                    ((isinstance(old_ncomm_factor, Operator) and
                      ncomm_factor != old_ncomm_factor) or
                     isinstance(old_ncomm_factor, Pow)):
                break
            if is_adjoint(ncomm_factor):
                if not is_adjoint(old_ncomm_factor) or \
                         ncomm_factor != old_ncomm_factor:
                    break
            else:
                if not isinstance(ncomm_factor, Pow):
                    if is_adjoint(old_ncomm_factor):
                        break
                else:
                    if isinstance(old_ncomm_factor, Pow):
                        old_base = old_ncomm_factor.base
                        old_degree = old_ncomm_factor.exp
                    else:
                        old_base = old_ncomm_factor
                        old_degree = 1
                    if old_base != ncomm_factor.base:
                        break
                    if old_degree > ncomm_factor.exp:
                        break
                    if old_degree < ncomm_factor.exp:
                        if j != len(old_ncomm_factors) - 1:
                            if j != 0:
                                break
                            else:
                                left_remainder = old_base ** (
                                    ncomm_factor.exp - old_degree)
                        else:
                            right_remainder = old_base ** (
                                ncomm_factor.exp - old_degree)
        else:
            new_monomial = 1
            for var in new_var_list:
                new_monomial *= var
            new_monomial *= left_remainder * new_sub * right_remainder
            for j in range(i + len(old_ncomm_factors), len(ncomm_factors)):
                new_monomial *= ncomm_factors[j]
            new_monomial *= comm_monomial
            break
        new_var_list.append(ncomm_factors[i])
    else:
        if not is_constant_term and comm_factors != ():
            new_monomial = comm_monomial
            for factor in ncomm_factors:
                new_monomial *= factor
        else:
            return monomial
    if not isinstance(new_sub, (float, int, complex)) and new_sub.is_Add:
        return expand(new_monomial)
    else:
        return new_monomial