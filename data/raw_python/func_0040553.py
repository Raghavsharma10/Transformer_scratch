def get_product_version(path: typing.Union[str, Path]) -> VersionInfo:
    """
    Get version info from executable

    Args:
        path: path to the executable

    Returns: VersionInfo
    """
    path = Path(path).absolute()
    pe_info = pefile.PE(str(path))

    try:
        for file_info in pe_info.FileInfo:  # pragma: no branch
            if isinstance(file_info, list):
                result = _parse_file_info(file_info)
                if result:
                    return result
            else:
                result = _parse_file_info(pe_info.FileInfo)
                if result:
                    return result

        raise RuntimeError(f'unable to obtain version from {path}')
    except (KeyError, AttributeError) as exc:
        traceback.print_exc()
        raise RuntimeError(f'unable to obtain version from {path}') from exc