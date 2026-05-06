def cellsim(self, cellindex, return_just_cell = False):
        """
        Do the actual simulations of LFP, using synaptic spike times from
        network simulation.


        Parameters
        ----------
        cellindex : int
            cell index between 0 and population size-1.
        return_just_cell : bool
            If True, return only the `LFPy.Cell` object
            if False, run full simulation, return None.


        Returns
        -------
        None or `LFPy.Cell` object


        See also
        --------
        hybridLFPy.csd, LFPy.Cell, LFPy.Synapse, LFPy.RecExtElectrode
        """
        tic = time()
        
        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            #with several cells, NEURON can only hold one cell at the time
            allsecnames = []
            allsec = []
            for sec in cell.allseclist:
                allsecnames.append(sec.name())
                for seg in sec:
                    allsec.append(sec.name())
            cell.allsecnames = allsecnames
            cell.allsec = allsec
            return cell
        else:
            self.insert_all_synapses(cellindex, cell)

            #electrode object where LFPs are calculated
            electrode = LFPy.RecExtElectrode(**self.electrodeParams)

            if self.calculateCSD:
                cell.tvec = np.arange(cell.totnsegs)
                cell.imem = np.eye(cell.totnsegs)
                csdcoeff = csd.true_lam_csd(cell,
                                self.populationParams['radius'], electrode.z)
                csdcoeff *= 1E6 #nA mum^-3 -> muA mm^-3 conversion
                del cell.tvec, cell.imem
                cell.simulate(electrode, dotprodcoeffs=[csdcoeff],
                              **self.simulationParams)
                cell.CSD = helpers.decimate(cell.dotprodresults[0],
                                            q=self.decimatefrac)
            else:
                cell.simulate(electrode,
                              **self.simulationParams)

            cell.LFP = helpers.decimate(electrode.LFP,
                                        q=self.decimatefrac)


            cell.x = electrode.x
            cell.y = electrode.y
            cell.z = electrode.z

            cell.electrodecoeff = electrode.electrodecoeff

            #put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if type(attr) == np.ndarray:
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                              time()-tic))