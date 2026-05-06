def addHydrogens(molecule, usedPyroles=None):
    """(molecule) -> add implicit hydrogens to a molecule.
    If the atom has specified valences and the atom must be
    charged then a Valence Error is raised"""
    for atom in molecule.atoms:
        # if the atom has an explicit hcount, we can't set the
        # hcount
        if atom.has_explicit_hcount:
            atom.hcount = atom.explicit_hcount
            continue
        
        if atom.valences:            
            for valence in atom.valences:
                hcount = max(0, int(valence - atom.sumBondOrders() + atom.charge))
                if hcount >= 0:
                    break
            else:
                if usedPyroles and not usedPyroles.has_key(atom.handle):
                    #print atom.symbol, atom.valences, atom.hcount, atom.charge,\
                    #      atom.sumBondOrders()
                    #print [x.bondtype for x in atom.bonds]
                    #print molecule.cansmiles()
                    raise PinkyError("Valence error in atom %s"%molecule.atoms.index(atom))
                pass

            #hcount = int(hcount)
            atom.hcount = hcount
    return molecule