def run(self,candidates,parameters):
        """
        Run simulation for each candidate
        
        This run method will loop through each candidate and run the simulation
        corresponding to its parameter values. It will populate an array called
        traces with the resulting voltage traces for the simulation and return it.
        """

        traces = []
        start_time = time.time()
        
        if self.num_parallel_evaluations == 1:
            
            for candidate_i in range(len(candidates)):
                
                candidate = candidates[candidate_i]
                sim_var = dict(zip(parameters,candidate))
                pyneuroml.pynml.print_comment_v('\n\n  - RUN %i (%i/%i); variables: %s\n'%(self.count,candidate_i+1,len(candidates),sim_var))
                self.count+=1
                t,v = self.run_individual(sim_var)
                traces.append([t,v])

        else:
            import pp
            ppservers = ()
            job_server = pp.Server(self.num_parallel_evaluations, ppservers=ppservers)
            pyneuroml.pynml.print_comment_v('Running %i candidates across %i local processes'%(len(candidates),job_server.get_ncpus()))
            jobs = []
            
            for candidate_i in range(len(candidates)):
                
                candidate = candidates[candidate_i]
                sim_var = dict(zip(parameters,candidate))
                pyneuroml.pynml.print_comment_v('\n\n  - PARALLEL RUN %i (%i/%i of curr candidates); variables: %s\n'%(self.count,candidate_i+1,len(candidates),sim_var))
                self.count+=1
                cand_dir = self.generate_dir+"/CANDIDATE_%s"%candidate_i
                if not os.path.exists(cand_dir):
                    os.mkdir(cand_dir)
                vars = (sim_var,
                        self.ref,
                        self.neuroml_file,
                        self.nml_doc,
                        self.still_included,
                        cand_dir,
                        self.target,
                        self.sim_time,
                        self.dt,
                        self.simulator)
                        
                job = job_server.submit(run_individual, vars, (), ("pyneuroml.pynml",'pyneuroml.tune.NeuroMLSimulation','shutil','neuroml'))
                jobs.append(job)
            
            for job_i in range(len(jobs)):
                job = jobs[job_i]
                pyneuroml.pynml.print_comment_v("Checking parallel job %i/%i; set running so far: %i"%(job_i,len(jobs),self.count))
                t,v = job()
                traces.append([t,v])
                
                #pyneuroml.pynml.print_comment_v("Obtained: %s"%result) 
                
            ####job_server.print_stats()
            job_server.destroy()
            print("-------------------------------------------")
                
                
            
        end_time = time.time()
        tot = (end_time-start_time)
        pyneuroml.pynml.print_comment_v('Ran %i candidates in %s seconds (~%ss per job)'%(len(candidates),tot,tot/len(candidates)))

        return traces