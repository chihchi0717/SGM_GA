import math as m

def draw_tri(sid, ang):
    import math as m

    S1, S2 = sid
    A1 = ang[0]  # degrees
    rad = 180 / m.pi

    if S1 <= 0 or S2 <= 0 or A1 <= 1 or A1 >= 178:
        return None

    try:
        S3 = (S1**2 + S2**2 - 2*S1*S2*m.cos(A1 / rad))**0.5
    except Exception as e:
        print(f"❌ 計算 S3 時失敗: {e}")
        return None

    # 三角不等式檢查
    if not (S1 + S2 > S3 and S1 + S3 > S2 and S2 + S3 > S1):
        return None

    try:
        cosA = (S1**2 + S3**2 - S2**2) / (2 * S1 * S3)
        if abs(cosA) > 1:
            return None
        A2 = m.acos(cosA) / rad
    except Exception as e:
        print(f"❌ 計算 A2 時失敗: {e}")
        return None

    # 計算 vx, vy
    vx = [S1 * m.cos(0), S2 * m.cos(A1 / rad)]
    vy = [S1 * m.sin(0), S2 * m.sin(A1 / rad)]
    return vx, vy

    
def draw_(sid_ang, changeable_side, changeable_angle, generation, num, mode, initial):

    sid = sid_ang[:changeable_side+1]
    ang = sid_ang[changeable_side+1:]

    assert len(sid) == 2 and len(ang) == 1, "只支援三角形"

    result = draw_tri(sid, ang)
    if result is None:
        return 0, [], []
    vx, vy = result


    for i in range(len(sid)):
        sid[i] = sid[i] 

    x = [vx[0], 0]
    y = [vy[0], 0]

    for i in range(1, len(vx)):
        x.append(x[i] + vx[i])
        y.append(y[i] + vy[i])

    return 1, x, y


