def addStencilBranch(self, disp, weight):
        """
        Set or overwrite the stencil weight for the given direction
        @param disp displacement vector
        @param weight stencil weight
        """
        self.stencil[tuple(disp)] = weight
        self.__setPartionLogic(disp)