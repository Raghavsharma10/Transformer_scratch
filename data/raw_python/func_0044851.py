def add_post_process(self, name, post_process, description=""):
        """add a post-process

        Parameters
        ----------
        name : str
            name of the post-traitment
        post_process : callback (function of a class with a __call__ method
                                 or a streamz.Stream).
            this callback have to accept the simulation state as parameter
            and return the modifield simulation state.
            if a streamz.Stream is provided, it will me plugged_in with the
            previous streamz (and ultimately to the initial_stream). All these
            stream accept and return the simulation state.
        description : str, optional, Default is "".
            give extra information about the post-processing
        """

        self._pprocesses.append(PostProcess(name=name,
                                            function=post_process,
                                            description=description))
        self._pprocesses[-1].function(self)