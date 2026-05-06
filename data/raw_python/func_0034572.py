def get_calculator_impstr(calculator_name):
    """
    Returns the import string for the calculator
    """
    if calculator_name.lower() == "gpaw" or calculator_name is None:
        return "from gpaw import GPAW as custom_calculator"
    elif calculator_name.lower() == "espresso":
        return "from espresso import espresso as custom_calculator"
    else:
        possibilities = {"abinit":"abinit.Abinit",
                         "aims":"aims.Aims",
                         "ase_qmmm_manyqm":"AseQmmmManyqm",
                         "castep":"Castep",
                         "dacapo":"Dacapo",
                         "dftb":"Dftb",
                         "eam":"EAM",
                         "elk":"ELK",
                         "emt":"EMT",
                         "exciting":"Exciting",
                         "fleur":"FLEUR",
                         "gaussian":"Gaussian",
                         "gromacs":"Gromacs",
                         "mopac":"Mopac",
                         "morse":"MorsePotential",
                         "nwchem":"NWChem",
                         'siesta':"Siesta",
                         "tip3p":"TIP3P",
                         "turbomole":"Turbomole",
                         "vasp":"Vasp",
                         }
        
        current_val = possibilities.get(calculator_name.lower())
        
        package, class_name = (calculator_name,current_val) if current_val else calculator_name.rsplit('.',1)
        
        return "from ase.calculators.{} import {} as custom_calculator".format(package, class_name)