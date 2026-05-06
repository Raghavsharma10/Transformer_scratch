def _balance(self, ):
        """ calc unbalanced charges and radicals for skin atoms
        """
        meta = h.meta
        for n in (skin_reagent.keys() | skin_product.keys()):
            lost = skin_reagent[n]
            cycle_lost = cycle(lost)
            new = skin_product[n]
            cycle_new = cycle(new)
            atom = h._node[n]
            dr = atom.p_radical - atom.radical
            # radical balancing
            if dr > 0:  # radical added or increased.
                for _, m in zip(range(dr), cycle_lost):  # homolysis
                    s_atom = h._node[m]
                    s_atom.p_multiplicity = radical_unmap[s_atom.p_radical + 1]
                    meta.setdefault('rule #14. atom lost. common atom radical added or increased. '
                                    'lost atom radical added', []).append((m, n))
                for m in lost[dr:]:
                    meta.setdefault('rule #15. atom lost. common atom radical added or increased. '
                                    'lost atom radical unchanged', []).append((m, n))
            elif dr < 0:  # radical removed or decreased.
                if n in skin_product:
                    for m in lost:
                        meta.setdefault('rule #20. atom lost. common atom radical removed or decreased. '
                                        'lost atom radical unchanged', []).append((m, n))
                else:
                    for _, m in zip(range(-dr), cycle_lost):  # radical elimination
                        s_atom = h._node[m]
                        s_atom.p_multiplicity = radical_unmap[s_atom.p_radical + 1]
                        meta.setdefault('rule #21. atom lost. common atom radical removed or decreased. '
                                        'lost atom radical added', []).append((m, n))
                    for m in lost[-dr:]:
                        meta.setdefault('rule #20. atom lost. common atom radical removed or decreased. '
                                        'lost atom radical unchanged', []).append((m, n))
            else:
                env = h.environment(n)
                sv = atom.get_valence([(b.reagent, a.reagent) for b, a in env if b.order])
                pv = atom.p_get_valence([(b.product, a.product) for b, a in env if b.p_order])
                sh, ph = h.atom_total_h(n)

                dv = pv - sv
                dh = ph - sh
                dc = atom.p_charge - atom.charge

                if not (dv or dh or dc):  # common atom unchanged. Substitution, Elimination
                    for m in skins:
                        meta.setdefault('rule #1. atom lost. common atom unchanged. '
                                        'substitution, elimination, addition', []).append((m, n))
                elif dv == dh == dc < 0:  # explicit hydrogen removing
                    for m in skins:
                        h._node[m].p_charge = 1
                        meta.setdefault('rule #4. atom lost. common atom deprotonation', []).append((m, n))
                else:
                    for m in skins:
                        meta.setdefault('rule #5. atom lost. common atom changed. '
                                        'convert to reduction or oxidation', []).append((m, n))

                    pth = ph + sum(h.atom_total_h(x)[1] for x in skins)
                    if n in skin_product:
                        sth = sh + sum(h.atom_total_h(x)[0] for x in skin_product[n])
                    else:
                        sth = sh
                    dth = pth - sth

        for n, skins in skin_product.items():
            cycle_skins = cycle(skins)
            atom = h._node[n]
            dr = atom.p_radical - atom.radical
            # radical balancing
            if dr > 0:  # radical added or increased.
                if n in skin_reagent:
                    for m in skins:
                        meta.setdefault('rule #16. atom new. common atom radical added or increased. '
                                        'new atom radical unchanged', []).append((m, n))
                else:
                    for _, m in zip(range(dr), cycle_skins):  # radical addition
                        s_atom = h._node[m]
                        s_atom.multiplicity = radical_unmap[s_atom.radical + 1]
                        meta.setdefault('rule #17. atom new. common atom radical added or increased. '
                                        'new atom radical added', []).append((m, n))
                    for m in skins[dr:]:
                        meta.setdefault('rule #16. atom new. common atom radical added or increased. '
                                        'new atom radical unchanged', []).append((m, n))
            elif dr < 0:  # radical removed or decreased.
                for _, m in zip(range(-dr), cycle_skins):  # recombination
                    s_atom = h._node[m]
                    s_atom.multiplicity = radical_unmap[s_atom.radical + 1]
                    meta.setdefault('rule #18. atom new. common atom radical removed or decreased. '
                                    'new atom radical added', []).append((m, n))
                for m in skins[-dr:]:
                    meta.setdefault('rule #19. atom new. common atom radical removed or decreased. '
                                    'new atom radical unchanged', []).append((m, n))
            else:
                env = h.environment(n)
                sv = atom.get_valence([(b.reagent, a.reagent) for b, a in env if b.order])
                pv = atom.p_get_valence([(b.product, a.product) for b, a in env if b.p_order])
                sh, ph = h.atom_total_h(n)

                dv = pv - sv
                dh = ph - sh
                dc = atom.p_charge - atom.charge

                if not (dv or dh or dc):  # common atom unchanged. Substitution, Addition
                    for m in skins:
                        meta.setdefault('rule #2. atom new. common atom unchanged. '
                                        'substitution, elimination, addition', []).append((m, n))
                elif dv == dh == dc > 0:  # explicit hydrogen addition
                    for m in skins:
                        h._node[m].charge = 1
                        h.meta.setdefault('rule #3. atom new. common atom protonation', []).append((m, n))
                else:
                    for m in skins:
                        meta.setdefault('rule #6. atom new. common atom changed. '
                                        'convert to reduction or oxidation', []).append((m, n))

                    sth = sh + sum(h.atom_total_h(x)[0] for x in skins)
                    if n in skin_reagent:
                        pth = ph + sum(h.atom_total_h(x)[1] for x in skin_reagent[n])
                    else:
                        pth = ph
                    dth = pth - sth

        for n, sp in reverse_ext.items():

            # charge neutralization
            if dc > 0:
                for _ in range(dc):
                    h.meta.setdefault('rule #7. charge neutralization. hydroxide radical added',
                                      []).append(h.add_atom(O(multiplicity=2), O(charge=-1)))
            elif dc < 0:
                for _ in range(-dc):
                    h.meta.setdefault('rule #8. charge neutralization. hydrogen radical added',
                                      []).append(h.add_atom(H(multiplicity=2), H(charge=1)))

            # hydrogen balancing
            if dth > 0:
                red_e = 0
                for m in sp['products']:
                    if h.nodes[m]['element'] == 'H':  # set reduction H if explicit H count increased
                        h.nodes[m]['s_radical'] = 2
                        red_e += 1
                        h.meta.setdefault('rule #11. protonation. new explicit hydrogen radical added',
                                          []).append(m)

                red = []
                for _ in range(dth - red_e):  # add reduction agents
                    m = h.add_atom(H(multiplicity=2), H())
                    red.append(m)
                    h.meta.setdefault('rule #10. protonation. hydrogen radical added', []).append(m)
                red = iter(red)

                dih = sub(*h.atom_implicit_h(n))
                if dih < 0:  # attach reduction H to central atom if implicit H atoms count increased
                    for _ in range(-dih):
                        m = next(red)
                        h.add_bond(m, n, None)
                        h.meta.setdefault('rule #12. protonation. new implicit hydrogen radical added',
                                          []).append(m)

                for m in sp['reagents']:  # attach reduction H if detached group implicit H count increased
                    dih = sub(*h.atom_implicit_h(m))
                    if dih < 0:
                        for _ in range(-dih):
                            o = next(red)
                            h.add_bond(o, m, None)
            elif dth < 0:
                oxo = []
                for _ in range(-dth):
                    m = h.add_atom(O(multiplicity=2), O())
                    oxo.append(m)
                    h.meta.setdefault('rule #9. deprotonation. hydroxide radical added', []).append(m)
                oxo = iter(oxo)

                for m in sp['reagents']:
                    if h.nodes[m]['element'] == 'H':
                        o = next(oxo)
                        h.add_bond(o, m, None)
                        h.meta.setdefault('rule #13. hydrogen accepting by hydroxide radical added',
                                          []).append(m)

        return h