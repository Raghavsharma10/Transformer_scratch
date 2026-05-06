def run_individual(sim_var, 
                   reference,
                   neuroml_file,
                   nml_doc,
                   still_included,
                   generate_dir,
                   target,
                   sim_time, 
                   dt, 
                   simulator,
                   cleanup = True,
                   show=False):
    """
    Run an individual simulation.

    The candidate data has been flattened into the sim_var dict. The
    sim_var dict contains parameter:value key value pairs, which are
    applied to the model before it is simulated.

    """
 
    for var_name in sim_var.keys():
        
        individual_var_names = var_name.split('+')
        
        for individual_var_name in individual_var_names:
            words = individual_var_name.split('/')
            type, id1 = words[0].split(':')
            if ':' in words[1]:
                variable, id2 = words[1].split(':')
            else:
                variable = words[1]
                id2 = None

            units = words[2]
            value = sim_var[var_name]

            pyneuroml.pynml.print_comment_v('  Changing value of %s (%s) in %s (%s) to: %s %s'%(variable, id2, type, id1, value, units))

            if type == 'channel':
                channel = nml_doc.get_by_id(id1)
                
                if channel:
                    print("Setting channel %s"%(channel))
                    if variable == 'vShift':
                        channel.v_shift = '%s %s'%(value, units)
                else:
                    
                    pyneuroml.pynml.print_comment_v('Could not find channel with id %s from expression: %s'%(id1, individual_var_name))
                    exit()
                    

            elif type == 'cell':
                cell = None
                for c in nml_doc.cells:
                    if c.id == id1:
                        cell = c

                if variable == 'channelDensity':

                    chanDens = None
                    for cd in cell.biophysical_properties.membrane_properties.channel_densities + cell.biophysical_properties.membrane_properties.channel_density_v_shifts:
                        if cd.id == id2:
                            chanDens = cd

                    chanDens.cond_density = '%s %s'%(value, units)
                    
                elif variable == 'vShift_channelDensity':

                    chanDens = None
                    for cd in cell.biophysical_properties.membrane_properties.channel_density_v_shifts:
                        if cd.id == id2:
                            chanDens = cd

                    chanDens.v_shift = '%s %s'%(value, units)

                elif variable == 'channelDensityNernst':

                    chanDens = None
                    for cd in cell.biophysical_properties.membrane_properties.channel_density_nernsts:
                        if cd.id == id2:
                            chanDens = cd

                    chanDens.cond_density = '%s %s'%(value, units)

                elif variable == 'erev_id': # change all values of erev in channelDensity elements with only this id

                    chanDens = None
                    for cd in cell.biophysical_properties.membrane_properties.channel_densities + cell.biophysical_properties.membrane_properties.channel_density_v_shifts:
                        if cd.id == id2:
                            chanDens = cd

                    chanDens.erev = '%s %s'%(value, units)

                elif variable == 'erev_ion': # change all values of erev in channelDensity elements with this ion

                    chanDens = None
                    for cd in cell.biophysical_properties.membrane_properties.channel_densities + cell.biophysical_properties.membrane_properties.channel_density_v_shifts:
                        if cd.ion == id2:
                            chanDens = cd

                    chanDens.erev = '%s %s'%(value, units)

                elif variable == 'specificCapacitance': 

                    specCap = None
                    for sc in cell.biophysical_properties.membrane_properties.specific_capacitances:
                        if (sc.segment_groups == None and id2 == 'all') or sc.segment_groups == id2 :
                            specCap = sc

                    specCap.value = '%s %s'%(value, units)

                elif variable == 'resistivity': 

                    resistivity = None
                    for rs in cell.biophysical_properties.intracellular_properties.resistivities:
                        if (rs.segment_groups == None and id2 == 'all') or rs.segment_groups == id2 :
                            resistivity = rs

                    resistivity.value = '%s %s'%(value, units)

                else:
                    pyneuroml.pynml.print_comment_v('Unknown variable (%s) in variable expression: %s'%(variable, individual_var_name))
                    exit()

            elif type == 'izhikevich2007Cell':
                izhcell = None
                for c in nml_doc.izhikevich2007_cells:
                    if c.id == id1:
                        izhcell = c

                izhcell.__setattr__(variable, '%s %s'%(value, units))

            else:
                pyneuroml.pynml.print_comment_v('Unknown type (%s) in variable expression: %s'%(type, individual_var_name))



    new_neuroml_file =  '%s/%s'%(generate_dir,os.path.basename(neuroml_file))
    if new_neuroml_file == neuroml_file:
        pyneuroml.pynml.print_comment_v('Cannot use a directory for generating into (%s) which is the same location of the NeuroML file (%s)!'% \
                  (neuroml_file, generate_dir))

    pyneuroml.pynml.write_neuroml2_file(nml_doc, new_neuroml_file)

    for include in still_included:
        inc_loc = '%s/%s'%(os.path.dirname(os.path.abspath(neuroml_file)),include)
        pyneuroml.pynml.print_comment_v("Copying non included file %s to %s (%s) beside %s"%(inc_loc, generate_dir,os.path.abspath(generate_dir), new_neuroml_file))
        shutil.copy(inc_loc, generate_dir)
        
        

    from pyneuroml.tune.NeuroMLSimulation import NeuroMLSimulation

    sim = NeuroMLSimulation(reference, 
                         neuroml_file = new_neuroml_file,
                         target = target,
                         sim_time = sim_time, 
                         dt = dt, 
                         simulator = simulator, 
                         generate_dir = generate_dir,
                         cleanup = cleanup,
                         nml_doc = nml_doc)

    sim.go()

    if show:
        sim.show()

    return sim.t, sim.volts