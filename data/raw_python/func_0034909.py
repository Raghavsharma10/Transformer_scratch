def XstarT_dot(self,M):
        """ get dot product of Xhat and M """
        if 0:
            #TODO: implement this properly
            pass
        else:
            RV = np.dot(self.Xstar().T,M)
        return RV