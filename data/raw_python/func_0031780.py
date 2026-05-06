def _compute_J(self):
        '''
        Compute the current amplitude corresponding to the exponential
        synapse model PSP amplitude
        
        Derivation using sympy:
        ::
            from sympy import *
            #define symbols
            t, tm, Cm, ts, Is, Vmax = symbols('t tm Cm ts Is Vmax')
            
            #assume zero delay, t >= 0
            #using eq. 8.10 in Sterrat et al
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V = %s' % V
            
            #find time of V == Vmax
            dVdt = diff(V, t)
            print 'dVdt = %s' % dVdt
            
            [t] = solve(dVdt, t)
            print 't(t@dVdT==Vmax) = %s' % t
            
            #solve for Is at time of maxima
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V(%s) = %s' % (t, V)
            
            [Is] = solve(V-Vmax, Is)
            print 'Is = %s' % Is
        
        resulting in:
        ::
            Cm*Vmax*(-tm + ts)/(tm*ts*(exp(tm*log(ts/tm)/(tm - ts))
                                     - exp(ts*log(ts/tm)/(tm - ts))))
        
        '''
        #LIF params
        tm = self.model_params['tau_m']
        Cm = self.model_params['C_m']
        
        #synapse
        ts = self.model_params['tau_syn_ex']
        Vmax = self.PSP_e
        
        #max current amplitude
        J = Cm*Vmax*(-tm + ts)/(tm*ts*(np.exp(tm*np.log(ts/tm)/(tm - ts))
                                     - np.exp(ts*np.log(ts/tm)/(tm - ts))))
        
        #unit conversion pF*mV -> nA
        J *= 1E-3
        
        return J