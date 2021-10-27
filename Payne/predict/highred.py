import numpy as np

class highAv(object):
    """docstring for highAv"""
    def __init__(self, filters):
        super(highAv, self).__init__()

        AvTab = np.genfromtxt(
            'Avoutputpars.dat',
            dtype=('S10',float,float,float,float,float),skip_header=1,
            names=('filter','a1','b1','a2','b2','c2'))

        self.Avlist = []
        for ff in filters:
            AvTab_i = AvTab[np.in1d(t['filter'],bytes(ff,'ascii'))]
            self.Avlist.append([AvTab_i['a1'],AvTab_i['b1'],AvTab_i['a2'],AvTab_i['b2'],AvTab_i['c2']])

    def getAvaprox(self,Av,Rv,pars):
        a1,b1,a2,b2,c2 = pars
        return a1 + b1 * Av * (a2 + b2 * Rv + c2 * Rv**2.0)  

    def calc(self,BC0,Av,Rv):

        offset = [getAvaprox(Av,Rv,pars_i) for pars_i in self.Avlist]

        return np.array([offset_i + BC0_i for offset_i,BC0_i in zip(offset,BC0)])
