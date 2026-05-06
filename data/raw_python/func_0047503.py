def read_sbml(filename):
    """
    Read the model from a SBML file.

    :param filename: SBML filename to read the model from
    :return: A tuple, consisting of :class:`~means.core.model.Model` instance,
             set of parameter values, and set of initial conditions variables.
    """
    import libsbml

    if not os.path.exists(filename):
        raise IOError('File {0!r} does not exist'.format(filename))

    reader = libsbml.SBMLReader()
    document = reader.readSBML(filename)

    sbml_model = document.getModel()
    if not sbml_model:
        raise ValueError('Cannot parse SBML model from {0!r}'.format(filename))

    species = sympy.symbols([s.getId() for s in sbml_model.getListOfSpecies()])
    initial_conditions = [s.getInitialConcentration() for s in sbml_model.getListOfSpecies()]
    compartments = sympy.symbols([s.getId() for s in sbml_model.getListOfCompartments()])
    compartment_sizes = [s.getSize() for s in sbml_model.getListOfCompartments()]

    reactions = map(_parse_reaction, sbml_model.getListOfReactions())

    # getListOfParameters is an attribute of the model for SBML Level 1&2
    parameters_with_values = [(sympy.Symbol(p.getId()), p.getValue())
                              for p in sbml_model.getListOfParameters()]
    parameter_values = dict(parameters_with_values)
    parameters = map(lambda x: x[0], parameters_with_values)
    if not parameters:
        track_local_parameters = True
        parameters = set()
        parameter_values = {}
    else:
        track_local_parameters = False


    stoichiometry_matrix = np.zeros((len(species), len(reactions)), dtype=int)
    propensities = []
    for reaction_index, reaction in enumerate(reactions):
        if track_local_parameters:
            for param, value in reaction.parameters:
                parameters.add(param)
                parameter_values[param] = value
        reactants = reaction.reactants
        products = reaction.products
        propensities.append(reaction.propensity)
        for species_index, species_id in enumerate(species):
            net_stoichiometry = products.get(species_id, 0) - reactants.get(species_id, 0)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry


    if track_local_parameters:
        # sympy does not allow sorting its parameter lists by default,
        # explicitly tell to sort by str representation
        sorted_parameters = sorted(parameters, key=str)
    else:
        sorted_parameters = parameters

    parameter_values_list = [parameter_values[p] for p in sorted_parameters]

    # We need to concatenate compartment names and parameters as in our framework we cannot differentiate the two
    compartments_and_parameters = compartments + sorted_parameters
    parameter_values_list = compartment_sizes + parameter_values_list

    model = Model(species, compartments_and_parameters, propensities, stoichiometry_matrix)

    return model, parameter_values_list, initial_conditions