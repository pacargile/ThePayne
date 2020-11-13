import jax.numpy as np

def chebval(x, c, tensor=True):
    """
    Evaluate a Chebyshev series at points x.
    If `c` is of length `n + 1`, this function returns the value:
    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)
    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.
    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).
    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.
    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.
        .. versionadded:: 1.7.0
    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.
    See Also
    --------
    chebval2d, chebgrid2d, chebval3d, chebgrid3d
    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.
    Examples
    --------
    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

def polycalc(coef,inwave):
     # define obs wave on normalized scale
     x = inwave - inwave.min()
     x = 2.0*(x/x.max())-1.0
     # build poly coef
     # c = np.insert(coef[1:],0,0)
     poly = chebval(x,coef)
     # epoly = np.exp(coef[0]+poly)
     # epoly = coef[0]+poly
     return poly

def airtovacuum(inwave):
     """
          Using the relationship from Ciddor (1996) and transcribed in Shetrone et al. 2015
     """
     a = 0.0
     b1 = 5.792105E-2
     b2 = 1.67917E-3
     c1 = 238.0185
     c2 = 57.362

     deltawave = a + (b1/(c1-(1.0/inwave**2.0))) + (b2/(c2-(1.0/inwave**2.0)))

     return inwave*(deltawave+1)
