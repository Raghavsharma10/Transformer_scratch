def linexp(value, in_min, in_max, out_min, out_max, clip=True):
        """
        maps value \in linear range [in_min,in_max] corresponding value \in exponential range [out_min, out_max]
        (extrapolating if needed)

        :param value: value to be mapped 
        :param in_min: input range minimum
        :param in_max: input range maximum
        :param out_min: what input range minimum is mapped to
        :param out_max: what input range maximum is mapped to
        :param clip: if True, the output value is clipped to range [out_min, out_max]        
        :return: mapping from value in linear input range to value in exponential output range (extrapolating if needed)
        """
        if out_min == 0:
            return None
        if in_min == in_max:
            if value == in_min and out_min == out_max:
                return out_min
            return None

        output = math.pow(out_max / out_min, (value - in_min) / (in_max - in_min)) * out_min
        if clip:
            output = Mapping.clip_value(output, out_min, out_max)
        return output