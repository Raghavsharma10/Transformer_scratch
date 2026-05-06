def dummyctrl(self,r,ctrl):
        """creates a DummyVertex at rank r inserted in the ctrl dict
           of the associated edge and layer.

           Arguments:
              r (int): rank value
              ctrl (dict): the edge's control vertices
           
           Returns:
              DummyVertex : the created DummyVertex.
        """
        dv = DummyVertex(r)
        dv.view.w,dv.view.h=self.dw,self.dh
        self.grx[dv] = dv
        dv.ctrl = ctrl
        ctrl[r] = dv
        self.layers[r].append(dv)
        return dv