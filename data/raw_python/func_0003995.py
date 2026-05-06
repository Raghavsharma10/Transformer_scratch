def load_pdb(filename):
    """Loads a single molecule from a pdb file.

       This function does support only a small fragment from the pdb specification.
       It assumes that there is only one molecular geometry in the pdb file.
    """
    with open(filename) as f:
        numbers = []
        coordinates = []
        occupancies = []
        betas = []
        for line in f:
            if line.startswith("ATOM"):
                symbol = line[76:78].strip()
                numbers.append(periodic[symbol].number)
                coordinates.append([float(line[30:38])*angstrom, float(line[38:46])*angstrom, float(line[46:54])*angstrom])
                occupancies.append(float(line[54:60]))
                betas.append(float(line[60:66]))
    if len(numbers) > 0:
        molecule = Molecule(numbers, coordinates)
        molecule.occupancies = np.array(occupancies)
        molecule.betas = np.array(betas)
        return molecule
    else:
        raise FileFormatError("No molecule found in pdb file %s" % filename)