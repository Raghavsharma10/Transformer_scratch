def sample(name, reads_in_tuple):
    """	Create a new sample.
	"""
    if name in [sample_x.get_name() for sample_x in __SAMPLES__]:
        rnftools.utils.error(
            "Multiple samples have the same name. Each sample must have a unique name.",
            program="RNFtools",
            subprogram="MIShmash",
            exception=ValueError,
        )

    Sample(
        name=name,
        reads_in_tuple=reads_in_tuple,
    )
    add_input(current_sample().fq_fns())