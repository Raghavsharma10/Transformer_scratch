def safe_shake(self, x, fun, fmax):
        '''Brings unknowns to the constraints, without increasing fun above fmax.

           Arguments:
            | ``x`` -- The unknowns.
            | ``fun`` -- The function being minimized.
            | ``fmax`` -- The highest allowed value of the function being
                          minimized.

           The function ``fun`` takes a mandatory argument ``x`` and an optional
           argument ``do_gradient``:
            | ``x``  --  the arguments of the function to be tested
            | ``do_gradient``  --  when False, only the function value is
                                   returned. when True, a 2-tuple with the
                                   function value and the gradient are returned
                                   [default=False]
        '''
        self.lock[:] = False
        def extra_equation(xx):
            f, g = fun(xx, do_gradient=True)
            return (f-fmax)/abs(fmax), g/abs(fmax)
        self.equations.append((-1,extra_equation))
        x, shake_counter, constraint_couter = self.free_shake(x)
        del self.equations[-1]
        return x, shake_counter, constraint_couter