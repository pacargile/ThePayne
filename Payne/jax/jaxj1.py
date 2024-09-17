import jax.numpy as np
from jax import jit
from jax import lax

RP = np.asarray([
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
    ],dtype=np.float32)

RQ = np.asarray([
    # 1.00000000000000000000E0,
    6.20836478118054335476E2,
    2.56987256757748830383E5,
    8.35146791431949253037E7,
    2.21511595479792499675E10,
    4.74914122079991414898E12,
    7.84369607876235854894E14,
    8.95222336184627338078E16,
    5.32278620332680085395E18,
    ],dtype=np.float32)

PP = np.asarray([
    7.62125616208173112003E-4,
    7.31397056940917570436E-2,
    1.12719608129684925192E0,
    5.11207951146807644818E0,
    8.42404590141772420927E0,
    5.21451598682361504063E0,
    1.00000000000000000254E0,
    ],dtype=np.float32)

PQ = np.asarray([
    5.71323128072548699714E-4,
    6.88455908754495404082E-2,
    1.10514232634061696926E0,
    5.07386386128601488557E0,
    8.39985554327604159757E0,
    5.20982848682361821619E0,
    9.99999999999999997461E-1,
    ],dtype=np.float32)

QP = np.asarray([
    5.10862594750176621635E-2,
    4.98213872951233449420E0,
    7.58238284132545283818E1,
    3.66779609360150777800E2,
    7.10856304998926107277E2,
    5.97489612400613639965E2,
    2.11688757100572135698E2,
    2.52070205858023719784E1,
    ],dtype=np.float32)

QQ = np.asarray([
    # 1.00000000000000000000E0,
    7.42373277035675149943E1,
    1.05644886038262816351E3,
    4.98641058337653607651E3,
    9.56231892404756170795E3,
    7.99704160447350683650E3,
    2.82619278517639096600E3,
    3.36093607810698293419E2,
    ],dtype=np.float32)

YP = np.asarray([
    1.26320474790178026440E9,
    -6.47355876379160291031E11,
    1.14509511541823727583E14,
    -8.12770255501325109621E15,
    2.02439475713594898196E17,
    -7.78877196265950026825E17,
    ],dtype=np.float32)

YQ = np.asarray([
    # 1.00000000000000000000E0, 
    5.94301592346128195359E2,
    2.35564092943068577943E5,
    7.34811944459721705660E7,
    1.87601316108706159478E10,
    3.88231277496238566008E12,
    6.20557727146953693363E14,
    6.87141087355300489866E16,
    3.97270608116560655612E18,
    ],dtype=np.float32)


Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1

THPIO4 = 2.35619449019234492885 
SQ2OPI = 0.79788456080286535587989

def j1(x):

    w = lax.map(lambda y: lax.cond(y <= 5.0,_j1a,_j1b,y),x)
    return w

# def _j1(x):

#     w = x
#     if (x < 0.0):
#         return -j1(-x)

#     if w <= 5.0:
#         z = x * x
#         w = _polevl(z,RP,3) / _p1evl(z, RQ, 8)
#         w = w * x * (z - Z1) * (z - Z2)
#         return w

#     w = 5.0 / x
#     z = w * w
#     p = _polevl(z, PP, 6) / _polevl(z, PQ, 6)
#     q = _polevl(z, QP, 7) / _p1evl(z, QQ, 7)
#     xn = x - THPIO4
#     p = p * cos(xn) - w * q * sin(xn)
#     w = (p * SQ2OPI / sqrt(x))
#     return w

def _j1a(x):
    w = x
    z = x * x
    w = _polevl(z,RP,3) / _p1evl(z, RQ, 8)
    w = w * x * (z - Z1) * (z - Z2)
    return w

def _j1b(x):
    w = 5.0 / x
    z = w * w
    p = _polevl(z, PP, 6) / _polevl(z, PQ, 6)
    q = _polevl(z, QP, 7) / _p1evl(z, QQ, 7)
    xn = x - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    w = (p * SQ2OPI / np.sqrt(x))
    return w

def _polevl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial
    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        try:
            ans += coef * x**power
        except OverflowError:
            pass
        power -= 1
    return ans


def _p1evl(x, coefs, N):
    """
    Port of cephes ``polevl.c``: evaluate polynomial, assuming coef[N] = 1
    See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
    """
    return _polevl(x, [1] + list(coefs), N)
