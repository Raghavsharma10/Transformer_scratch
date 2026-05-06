def _get_result_paths(self,data):
        """Return dict of {key: ResultPath}
        """

        #clustalw .aln is used when no or unkown output type specified
        _treeinfo_formats = {'nj':'.nj',
                            'dist':'.dst',
                            'nexus':'.tre'}

        result = {}
        par = self.Parameters
        abs = self._absolute

        if par['-align'].isOn():
            prefix = par['-infile'].Value.rsplit('.', 1)[0]
            #prefix = par['-infile'].Value.split('.')[0]
            aln_filename = self._aln_filename(prefix)
            if par['-newtree'].isOn():
                dnd_filename = abs(par['-newtree'].Value)
            elif par['-usetree'].isOn():
                dnd_filename = abs(par['-usetree'].Value)
            else:
                dnd_filename = abs(prefix + '.dnd')
            result['Align'] = ResultPath(Path=aln_filename,IsWritten=True)
            result['Dendro'] = ResultPath(Path=dnd_filename,IsWritten=True)
        elif par['-profile'].isOn():
            prefix1 = par['-profile1'].Value.rsplit('.', 1)[0]
            prefix2 = par['-profile2'].Value.rsplit('.', 1)[0]
            #prefix1 = par['-profile1'].Value.split('.')[0]
            #prefix2 = par['-profile2'].Value.split('.')[0]
            aln_filename = ''; aln_written = True
            dnd1_filename = ''; tree1_written = True
            dnd2_filename = ''; tree2_written = True
            aln_filename = self._aln_filename(prefix1)
            #usetree1
            if par['-usetree1'].isOn():
                tree1_written = False
            #usetree2
            if par['-usetree2'].isOn():
                tree2_written = False
            if par['-newtree1'].isOn():
                dnd1_filename = abs(par['-newtree1'].Value)
                aln_written=False
            else:
                dnd1_filename = abs(prefix1 + '.dnd')
            if par['-newtree2'].isOn():
                dnd2_filename = abs(par['-newtree2'].Value)
                aln_written=False
            else:
                dnd2_filename = abs(prefix2 + '.dnd')
            result['Align'] = ResultPath(Path=aln_filename,
                IsWritten=aln_written)
            result['Dendro1'] = ResultPath(Path=dnd1_filename,
                IsWritten=tree1_written)
            result['Dendro2'] = ResultPath(Path=dnd2_filename,
                IsWritten=tree2_written)
        elif par['-sequences'].isOn():
            prefix1 = par['-profile1'].Value.rsplit('.', 1)[0]
            prefix2 = par['-profile2'].Value.rsplit('.', 1)[0]
            #prefix1 = par['-profile1'].Value.split('.')[0] #alignment
            #prefix2 = par['-profile2'].Value.split('.')[0] #sequences
            aln_filename = ''; aln_written = True
            dnd_filename = ''; dnd_written = True

            aln_filename = self._aln_filename(prefix2)
            if par['-usetree'].isOn():
                dnd_written = False
            elif par['-newtree'].isOn():
                aln_written = False
                dnd_filename = abs(par['-newtree'].Value)
            else:
                dnd_filename = prefix2 + '.dnd'
            result['Align'] = ResultPath(Path=aln_filename,\
                IsWritten=aln_written)
            result['Dendro'] = ResultPath(Path=dnd_filename,\
                IsWritten=dnd_written)
        elif par['-tree'].isOn():
            prefix = par['-infile'].Value.rsplit('.', 1)[0]
            #prefix = par['-infile'].Value.split('.')[0]
            tree_filename = ''; tree_written = True
            treeinfo_filename = ''; treeinfo_written = False
            tree_filename = prefix + '.ph'
            if par['-outputtree'].isOn() and\
                par['-outputtree'].Value != 'phylip':
                treeinfo_filename = prefix +\
                    _treeinfo_formats[par['-outputtree'].Value]
                treeinfo_written = True
            result['Tree'] = ResultPath(Path=tree_filename,\
                IsWritten=tree_written)
            result['TreeInfo'] = ResultPath(Path=treeinfo_filename,\
                IsWritten=treeinfo_written)

        elif par['-bootstrap'].isOn():
            prefix = par['-infile'].Value.rsplit('.', 1)[0]
            #prefix = par['-infile'].Value.split('.')[0]
            boottree_filename = prefix + '.phb'
            result['Tree'] = ResultPath(Path=boottree_filename,IsWritten=True)

        return result