def compare_version(lhs_version, rhs_version):
    """
    <Major>.<Minor>.<Revision> 形式のバージョン文字列を比較する。

    :return:
        0<:	LHSがRHSより小さい
        0:	LHS == RHS
        0>:	LHSがRHSより大きい
    :rtype: int
    """

    lhs_major, lhs_minor, lhs_revision = [
        int(v) for v in lhs_version.split(".")]
    rhs_major, rhs_minor, rhs_revision = [
        int(v) for v in rhs_version.split(".")]

    if lhs_major < rhs_major:
        return -1
    elif lhs_major > rhs_major:
        return 1

    if lhs_minor < rhs_minor:
        return -1
    elif lhs_minor > rhs_minor:
        return 1

    if lhs_revision < rhs_revision:
        return -1
    elif lhs_revision > rhs_revision:
        return 1

    return 0