def to_meshlevel(meshcode):
    """メッシュコードから次数を算出する。

    Args:
        meshcode: メッシュコード
    Return:
        地域メッシュコードの次数
                1次(80km四方):1
                40倍(40km四方):40000
                20倍(20km四方):20000
                16倍(16km四方):16000
                2次(10km四方):2
                8倍(8km四方):8000
                5倍(5km四方):5000
                4倍(4km四方):4000
                2.5倍(2.5km四方):2500
                2倍(2km四方):2000
                3次(1km四方):3
                4次(500m四方):4
                5次(250m四方):5
                6次(125m四方):6
    """

    length = len(str(meshcode))
    if length == 4:
        return 1

    if length == 5:
        return 40000

    if length == 6:
        return 2

    if length == 7:
        if meshcode[6:7] in ['1','2','3','4']:
            return 5000

        if meshcode[6:7] == '6':
            return 8000

        if meshcode[6:7] == '5':
            return 20000

        if meshcode[6:7] == '7':
            return 16000

    if length == 8:
        return 3

    if length == 9:
        if meshcode[8:9] in ['1','2','3','4']:
            return 4

        if meshcode[8:9] == '5':
            return 2000

        if meshcode[8:9] == '6':
            return 2500

        if meshcode[8:9] == '7':
            return 4000

    if length == 10:
        if meshcode[9:10] in ['1','2','3','4']:
            return 5

    if length == 11:
        if meshcode[10:11] in ['1','2','3','4']:
            return 6

    raise ValueError('the meshcode is unsupported.')