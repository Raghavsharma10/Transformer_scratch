def update_hyperparameters(self, new_params, hyper_deriv_handling='default', exit_on_bounds=True, inf_on_error=True):
        r"""Update the kernel's hyperparameters to the new parameters.
        
        This will call :py:meth:`compute_K_L_alpha_ll` to update the state
        accordingly.
        
        Note that if this method crashes and the `hyper_deriv_handling` keyword
        was used, it may leave :py:attr:`use_hyper_deriv` in the wrong state.
        
        Parameters
        ----------
        new_params : :py:class:`Array` or other Array-like, length dictated by kernel
            New parameters to use.
        hyper_deriv_handling : {'default', 'value', 'deriv'}, optional
            Determines what to compute and return. If 'default' and
            :py:attr:`use_hyper_deriv` is True then the negative log-posterior
            and the negative gradient of the log-posterior with respect to the
            hyperparameters is returned. If 'default' and
            :py:attr:`use_hyper_deriv` is False or 'value' then only the negative
            log-posterior is returned. If 'deriv' then only the negative gradient
            of the log-posterior with respect to the hyperparameters is returned.
        exit_on_bounds : bool, optional
            If True, the method will automatically exit if the hyperparameters
            are impossible given the hyperprior, without trying to update the
            internal state. This is useful during MCMC sampling and optimization.
            Default is True (don't perform update for impossible hyperparameters).
        inf_on_error : bool, optional
            If True, the method will return `scipy.inf` if the hyperparameters
            produce a linear algebra error upon trying to update the Gaussian
            process. Default is True (catch errors and return infinity).
        
        Returns
        -------
        -1*ll : float
            The updated log posterior.
        -1*ll_deriv : array of float, (`num_params`,)
            The gradient of the log posterior. Only returned if
            :py:attr:`use_hyper_deriv` is True or `hyper_deriv_handling` is set
            to 'deriv'.
        """
        use_hyper_deriv = self.use_hyper_deriv
        if hyper_deriv_handling == 'value':
            self.use_hyper_deriv = False
        elif hyper_deriv_handling == 'deriv':
            self.use_hyper_deriv = True
        self.k.set_hyperparams(new_params[:len(self.k.free_params)])
        self.noise_k.set_hyperparams(
            new_params[len(self.k.free_params):len(self.k.free_params) + len(self.noise_k.free_params)]
        )
        if self.mu is not None:
            self.mu.set_hyperparams(
                new_params[len(self.k.free_params) + len(self.noise_k.free_params):]
            )
        self.K_up_to_date = False
        try:
            if exit_on_bounds:
                if scipy.isinf(self.hyperprior(self.params)):
                    raise GPImpossibleParamsError("Impossible values for params!")
            self.compute_K_L_alpha_ll()
        except Exception as e:
            if inf_on_error:
                if not isinstance(e, GPImpossibleParamsError) and self.verbose:
                    warnings.warn(
                        "Unhandled exception when updating GP! Exception was:\n%s\n"
                        "State of params is: %s"
                        % (traceback.format_exc(), str(self.free_params[:]))
                    )
                self.use_hyper_deriv = use_hyper_deriv
                if use_hyper_deriv and hyper_deriv_handling == 'default':
                    return (scipy.inf, scipy.zeros(len(self.free_params)))
                elif hyper_deriv_handling == 'deriv':
                    return scipy.zeros(len(self.free_params))
                else:
                    return scipy.inf
            else:
                self.use_hyper_deriv = use_hyper_deriv
                raise e
        self.use_hyper_deriv = use_hyper_deriv
        if use_hyper_deriv and hyper_deriv_handling == 'default':
            return (-1.0 * self.ll, -1.0 * self.ll_deriv)
        elif hyper_deriv_handling == 'deriv':
            return -1.0 * self.ll_deriv
        else:
            return -1.0 * self.ll