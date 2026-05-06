def _gen_input_mask(mask):
    """Generate input mask from bytemask"""
    return input_mask(
        shift=bool(mask & MOD_Shift),
        lock=bool(mask & MOD_Lock),
        control=bool(mask & MOD_Control),
        mod1=bool(mask & MOD_Mod1),
        mod2=bool(mask & MOD_Mod2),
        mod3=bool(mask & MOD_Mod3),
        mod4=bool(mask & MOD_Mod4),
        mod5=bool(mask & MOD_Mod5))