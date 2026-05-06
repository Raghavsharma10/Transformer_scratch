def to_alu_hlu_map(input_str):
    """Converter for alu hlu map

    Convert following input into a alu -> hlu map:
    Sample input:

    ```
      HLU Number     ALU Number
      ----------     ----------
        0               12
        1               23
    ```

    ALU stands for array LUN number
    hlu stands for host LUN number
    :param input_str: raw input from naviseccli
    :return: alu -> hlu map
    """
    ret = {}
    if input_str is not None:
        pattern = re.compile(r'(\d+)\s*(\d+)')
        for line in input_str.split('\n'):
            line = line.strip()
            if len(line) == 0:
                continue
            matched = re.search(pattern, line)
            if matched is None or len(matched.groups()) < 2:
                continue
            else:
                hlu = matched.group(1)
                alu = matched.group(2)
                ret[int(alu)] = int(hlu)
    return ret