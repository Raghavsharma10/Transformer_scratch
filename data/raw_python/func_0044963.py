def linlin(value, in_min, in_max, out_min, out_max, clip=True):
        """
        maps value \in [in_min,in_max] linearly to the corresponding value \in [out_min, out_max]
        (extrapolating if needed)
        e.g. linlin(0.3,0,1,10,20) = 13
            
        :param value: value to be mapped 
        :param in_min: input range minimum
        :param in_max: input range maximum
        :param out_min: what input range minimum is mapped to
        :param out_max: what input range maximum is mapped to
        :param clip: if True, the output value is clipped to range [out_min, out_max] 
        :return: linear mapping from value in input range to value in output range (extrapolating if needed)
         
         example: linlin(0.2, 0, 1, 10, 20) = 13
        """
        if in_min == in_max:
            if value == in_min and out_min == out_max:
                return out_min
            return None

        output = ((out_min + out_max) + (out_max - out_min) * (
            (2 * value - (in_min + in_max)) / float(in_max - in_min))) / 2.0
        if clip:
            output = Mapping.clip_value(output, out_min, out_max)
        return output