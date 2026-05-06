def get_validation_fields(self):
        '''get_validation_fields returns a list of tuples (each a field)
           we only require the exp_id to coincide with the folder name, for the sake
           of reproducibility (given that all are served from sample image or Github
           organization). All other fields are optional.
           To specify runtime variables, add to "experiment_variables"

                 0: not required, no warning
                 1: required, not valid
                 2: not required, warning      
                type: indicates the variable type
        '''
        return [("name",1,str),   # required
                ("time",1,int), 
                ("url",1,str), 
                ("description",1, str),
                ("instructions",1, str),
                ("exp_id",1,str),

                ("install",0, list),  # list of commands to install / build experiment 
                ("contributors",0, list), # not required
                ("reference",0, list), 
                ("cognitive_atlas_task_id",0,str),
                ("template",0,str)]