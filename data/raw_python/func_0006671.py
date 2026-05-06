def populate_summary_dict(self, genus=str(), key=str()):
        """
        :param genus: Non-supported genus to be added to the dictionary
        :param key: section of dictionary to be populated. Supported keys are: prediction, table, and results
        Populate self.summary_dict as required. If the genus is not provided, populate the dictionary for Salmonella
        Escherichia and Campylobacter. If the genus is provided, this genus is non-standard, and an 'empty' profile
        must be created for it
        """
        # If the genus is not provided, generate the generic dictionary
        if not genus:
            # Populate the summary dict
            self.summary_dict = {
                'Salmonella':
                    {
                        'prediction':
                            {
                                'header': 'Strain,Colitsin,Colistin,Spectinomycin,Quinolones,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Salmonella_prediction_summary.csv')
                            },
                        'table':
                            {
                                'header': 'Strain,parE,parC,gyrA,pmrB,pmrA,gyrB,16S_rrsD,23S,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Salmonella_table_summary.csv')
                            },
                        'results':
                            {
                                'header': 'Strain,Genus,Mutation,NucleotideChange,AminoAcidChange,Resistance,PMID,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'PointFinder_results_summary.csv')
                            }
                    },
                'Escherichia':
                    {
                        'prediction':
                            {
                                'header': 'Strain,Colistin,GentamicinC,gentamicinC,Streptomycin,Macrolide,Sulfonamide,'
                                          'Tobramycin,Neomycin,Fluoroquinolones,Aminocoumarin,Tetracycline,KanamycinA,'
                                          'Spectinomycin,B-lactamResistance,Paromomycin,Kasugamicin,Quinolones,G418,'
                                          'QuinolonesAndfluoroquinolones,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Escherichia_prediction_summary.csv')
                            },
                        'table':
                            {
                                'header': 'Strain,parE,parC,folP,gyrA,pmrB,pmrA,16S_rrsB,16S_rrsH,gyrB,ampC,'
                                          '16S_rrsC,23S,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Escherichia_table_summary.csv')
                            },
                        'results':
                            {
                                'header': 'Strain,Genus,Mutation,NucleotideChange,AminoAcidChange,Resistance,PMID,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'PointFinder_results_summary.csv')
                            }
                    },
                'Campylobacter':
                    {

                        'prediction':
                            {
                                'header': 'Strain,LowLevelIncreaseMIC,AssociatedWithT86Mutations,Macrolide,Quinolone,'
                                          'Streptinomycin,Erythromycin,IntermediateResistance,HighLevelResistance_'
                                          'nalidixic_and_ciprofloxacin,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Campylobacter_prediction_summary.csv')
                            },
                        'table':
                            {
                                'header': 'Strain,L22,rpsL,cmeR,gyrA,23S,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'Campylobacter_table_summary.csv')
                            },
                        'results':
                            {
                                'header': 'Strain,Genus,Mutation,NucleotideChange,AminoAcidChange,Resistance,PMID,\n',
                                'output': str(),
                                'summary': os.path.join(self.reportpath, 'PointFinder_results_summary.csv')
                            }
                    }
            }
        else:
            # Create the nesting structure as required
            if genus not in self.summary_dict:
                self.summary_dict[genus] = dict()
            if key not in self.summary_dict[genus]:
                self.summary_dict[genus][key] = dict()
            # The output section is the same regardless of the key
            self.summary_dict[genus][key]['output'] = str()
            # The results report is more generic, and contains all strains, so the header and summary are set to
            # the default values required to generate this report
            if key == 'results':
                self.summary_dict[genus][key]['header'] = \
                    'Strain,Genus,Mutation,NucleotideChange,AminoAcidChange,Resistance,PMID,\n'
                self.summary_dict[genus][key]['summary'] = \
                    os.path.join(self.reportpath, 'PointFinder_results_summary.csv')
            # Create an empty header, and a report with the genus name
            else:
                self.summary_dict[genus][key]['header'] = 'Strain,\n'
                self.summary_dict[genus][key]['summary'] = os.path.join(self.reportpath, '{genus}_{key}_summary.csv'
                                                                        .format(genus=genus,
                                                                                key=key))
                # Remove the report if it exists, as the script will append data to this existing report
                if os.path.isfile(self.summary_dict[genus][key]['summary']):
                    os.remove(self.summary_dict[genus][key]['summary'])