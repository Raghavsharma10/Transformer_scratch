def _yCellParams(self):
        '''
        Return dict with parameters for each population.
        The main operation is filling in cell type specific morphology
        '''
        #cell type specific parameters going into LFPy.Cell        
        yCellParams = {}
        for layer, morpho, _, _ in self.y_zip_list:
            yCellParams.update({layer : self.cellParams.copy()})
            yCellParams[layer].update({
                'morphology' : os.path.join(self.PATH_m_y, morpho),
            })
        return yCellParams