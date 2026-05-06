def _get_output_nodes(self, output_path, error_path):
        """
        Extracts output nodes from the standard output and standard error
        files.
        """
        from pymatgen.io.nwchem import NwOutput
        from aiida.orm.data.structure import StructureData
        from aiida.orm.data.array.trajectory import TrajectoryData

        ret_dict = []
        nwo = NwOutput(output_path)
        for out in nwo.data:
            molecules = out.pop('molecules', None)
            structures = out.pop('structures', None)
            if molecules:
                structlist = [StructureData(pymatgen_molecule=m)
                              for m in molecules]
                ret_dict.append(('trajectory',
                                 TrajectoryData(structurelist=structlist)))
            if structures:
                structlist = [StructureData(pymatgen_structure=s)
                              for s in structures]
                ret_dict.append(('trajectory',
                                 TrajectoryData(structurelist=structlist)))
            ret_dict.append(('output', ParameterData(dict=out)))

        # Since ParameterData rewrites it's properties (using _set_attr())
        # with keys from the supplied dictionary, ``source`` has to be
        # moved to another key. See issue #9 for details:
        # (https://bitbucket.org/epfl_theos/aiida_epfl/issues/9)
        nwo.job_info['program_source'] = nwo.job_info.pop('source', None)
        ret_dict.append(('job_info', ParameterData(dict=nwo.job_info)))
        
        return ret_dict