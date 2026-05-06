def _analyze_case(model_dir, bench_dir, config):
    """ Runs all of the verification checks on a particular case """
    bundle = livvkit.verification_model_module
    model_out = functions.find_file(model_dir, "*"+config["output_ext"])
    bench_out = functions.find_file(bench_dir, "*"+config["output_ext"])
    model_config = functions.find_file(model_dir, "*"+config["config_ext"])
    bench_config = functions.find_file(bench_dir, "*"+config["config_ext"])
    model_log = functions.find_file(model_dir, "*"+config["logfile_ext"])
    el = [
            bit_for_bit(model_out, bench_out, config),
            diff_configurations(model_config, bench_config, bundle, bundle),
            bundle.parse_log(model_log)
         ]
    return el