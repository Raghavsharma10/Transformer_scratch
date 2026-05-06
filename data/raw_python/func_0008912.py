def _assign_chunk(self, data, arr1, arr2, te, be, le, re, ovr, add=False):
        """
        Assign data from a chunk to the full array. The data in overlap regions
        will not be assigned to the full array

        Parameters
        -----------
        data : array
            Unused array (except for shape) that has size of full tile
        arr1 : array
            Full size array to which data will be assigned
        arr2 : array
            Chunk-sized array from which data will be assigned
        te : int
            Top edge id
        be : int
            Bottom edge id
        le : int
            Left edge id
        re : int
            Right edge id
        ovr : int
            The number of pixels in the overlap
        add : bool, optional
            Default False. If true, the data in arr2 will be added to arr1,
            otherwise data in arr2 will overwrite data in arr1
        """
        
        if te == 0:
            i1 = 0
        else:
            i1 = ovr
        
        if be == data.shape[0]:
            i2 = 0
            i2b = None
        else:
            i2 = -ovr
            i2b = -ovr
        
        if le == 0:
            j1 = 0
        else:
            j1 = ovr
        
        if re == data.shape[1]:
            j2 = 0
            j2b = None
        else:
            j2 = -ovr
            j2b = -ovr

        if add:
            arr1[te+i1:be+i2, le+j1:re+j2] += arr2[i1:i2b, j1:j2b]
        else:
            arr1[te+i1:be+i2, le+j1:re+j2] = arr2[i1:i2b, j1:j2b]