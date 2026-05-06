def main() -> None:
    """"Execute the main routine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", help="output directory", default=os.path.dirname(__file__))
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError("Output directory is missing: {}".format(outdir))

    for contracts in [0, 1, 5, 10]:
        if contracts == 0:
            pth = outdir / "functions_100_with_no_contract.py"
        elif contracts == 1:
            pth = outdir / "functions_100_with_1_contract.py"
        else:
            pth = outdir / "functions_100_with_{}_contracts.py".format(contracts)

        text = generate_functions(functions=100, contracts=contracts, disabled=False)
        pth.write_text(text)

    for contracts in [1, 5, 10]:
        if contracts == 1:
            pth = outdir / "functions_100_with_1_disabled_contract.py"
        else:
            pth = outdir / "functions_100_with_{}_disabled_contracts.py".format(contracts)

        text = generate_functions(functions=100, contracts=contracts, disabled=True)
        pth.write_text(text)

    for invariants in [0, 1, 5, 10]:
        if invariants == 0:
            pth = outdir / "classes_100_with_no_invariant.py"
        elif invariants == 1:
            pth = outdir / "classes_100_with_1_invariant.py"
        else:
            pth = outdir / "classes_100_with_{}_invariants.py".format(invariants)

        text = generate_classes(classes=100, invariants=invariants, disabled=False)
        pth.write_text(text)

    for invariants in [1, 5, 10]:
        if invariants == 1:
            pth = outdir / "classes_100_with_1_disabled_invariant.py"
        else:
            pth = outdir / "classes_100_with_{}_disabled_invariants.py".format(invariants)

        text = generate_classes(classes=100, invariants=invariants, disabled=True)
        pth.write_text(text)