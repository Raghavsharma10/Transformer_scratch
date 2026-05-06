def viewAngle(self, **kwargs):
        '''
        calculate view factor between one small and one finite surface
        vf =1/pi * integral(cos(beta1)*cos(beta2)/s**2) * dA
        according to VDI heatatlas 2010 p961
        '''
        v0 = self.cam2PlaneVectorField(**kwargs)
        # obj cannot be behind camera
        v0[2][v0[2] < 0] = np.nan

        _t, r = self.pose()
        n = self.planeSfN(r)
        # because of different x,y orientation:
        n[2] *= -1
#         beta2 = vectorAngle(v0, vectorToField(n) )
        beta2 = vectorAngle(v0, n)
        return beta2