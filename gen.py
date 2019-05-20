#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import argparse
import numpy as np
import os
import json

'''
generic spectral functions
'''

def gauss(x, A, mu, sigma):
    k = A
    return k * np.exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma))

def lorz(x, A, mu, gamma):
    k = 1
    return k / (1 + ((x-mu)/gamma)^2)

def lin(x, alpha, beta):
    return alpha*x + beta

def noise_normal(x):
    np.random.normal(0, np.sqrt(x), 1)

'''
custom spectral function
'''
def dblgauss(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return A1 * np.exp(-1*(x-mu1)*(x-mu1)/(2*sigma1*sigma1)) + A2 * np.exp(-1*(x-mu2)*(x-mu2)/(2*sigma2*sigma2))

def peakonslope(x, alpha, beta, A, mu, sigma):
    return alpha * x + beta + A * np.exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma))

'''
data generation
'''
class Args:
    def __init__(self, args):
        self.args = args

def gen_write_data(dirname, filename, genfct, args_obj):
    # model value
    x = np.linspace(-10, 10, 100) # start, stop, len
    args = args_obj.args
    y = genfct(x, *args)
    
    # add noise
    noise = np.vectorize(np.random.normal)(loc=0, scale=np.sqrt(y/100), size=1)
    err = (1/10)*np.sqrt(y)
    y = y + noise
    y[y<0] = 0 # get rid of negative values due to noise "undershoot"
    
    text = open("template").read()
    text = text.replace("%LEN%", str(len(x)))
    text = text.replace("%FILENAME%", filename)

    data = text.splitlines()
    for i in range(len(x)):
        data.append('%s %s %s 1' % (x[i], y[i], err[i]))
    
    # create folder (only necessary the first time, but...
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # save files
    open(os.path.join(dirname, filename), 'w').write('\n'.join(data))

def print_abspaths_json(dirname, fns):
    absdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dirname)
    copypaste = json.dumps( np.vectorize(os.path.join)(absdir, fns).tolist(), indent=2)
    print(copypaste)
    print()

def gen_01():
    print("three gauss's")

    dirname = "testdata"
    filenames = ["data01.dat", "data02.dat", "data03.dat"]
    fct = gauss
    args_lst = [Args((0.5, -4, 2, )), Args((1, 0, 2, )), Args((0.5, 6, 2, ))]
    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)

    print_abspaths_json(dirname, filenames)

def gen_02():
    print("2x3 datashape single gauss's")

    dirname = "test02"
    filenames = [["d11.dat", "d12.dat", "d13.dat"], ["d21.dat", "d22.dat", "d23.dat"]]
    fct = gauss
    args_lst = [
                [Args((1, -4, 0.5, )), Args((1, 0, 0.5, )), Args((1, 6, 0.5, ))],
                [Args((1, -10, 3, )), Args((1, 0, 8, )), Args((1, 10, 0.5, ))]
                ]
    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)

    print_abspaths_json(dirname, filenames)

def gen_03():
    print()
    
    dirname = "test03"
    filenames = [["d11.dat", "d12.dat", "d13.dat"], ["d21.dat", "d22.dat", "d23.dat"]]
    fct = dblgauss
    args_lst = [
                [Args((0.5, -4, 2, 0.1, 0, 2, )), Args((1, 0, 2, 0.3, -5, 2, )), Args((0.5, 6, 2, 1, 0, 2, ))],
                [Args((0.5, -10, 3, 1, 0, 2, )), Args((1, 0, 8, 1, 0, 2, )), Args((0.5, 10, 2, 1, 0, 2, ))]
                ]
    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)
    
    print_abspaths_json(dirname, filenames)

def gen_04():
    ''' ṕeak on gaussian background (single) '''
    dirname = "test04"
    filenames = ["d11.dat"]
    fct = dblgauss
    args_lst = [Args((0.4, -2, 6, 1, 3, 0.5, ))]
    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)
    
    print_abspaths_json(dirname, filenames)

def gen_05():
    ''' designed for separate background fitting '''
    dirname = "/home/jaga/source/ifitlab/iflproj/testdata/vect01"
    filenames = [["d11.dat", "d12.dat", "d13.dat"], ["d21.dat", "d22.dat", "d23.dat"], ["d31.dat", "d32.dat", "d33.dat"]]
    fct = dblgauss #  A1, mu1, sigma1, A2, mu2, sigma2
    args_lst = [
                [Args((0.4, -1, 6, 1, 5, 0.3, )), Args((0.4, -2, 6, 1, 2, 0.5, )), Args((0.4, -2, 6, 1, -3, 0.7, ))],
                [Args((0.5, -6, 8, 1.5, 4, 0.3, )), Args((0.5, -6, 8, 1.5, 1, 0.5, )), Args((0.5, -6, 8, 1.5, -2, 0.7, ))],
                [Args((0.6, -2, 7, 1, 1.5, 1, )), Args((0.6, -2, 7, 1, -0.5, 1, )), Args((0.6, -2, 7, 1, -2.1, 1, ))]
                ]

    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)
    
    print_abspaths_json(dirname, filenames)

def gen_06():
    ''' peak on sloping backgound (single) '''
    dirname = "test06"
    filenames = ["peakonslop.dat"]
    fct = peakonslope
    args_lst = [Args((0.1, 5, 7, -5, 0.5, ))]
    vgen = np.vectorize(gen_write_data)
    vgen(dirname, filenames, fct, args_lst)
    
    print_abspaths_json(dirname, filenames)


'''
main
'''
def main(args):
    logging.basicConfig(level=logging.INFO)
    
    print("do teh gen")

    #gen_01()
    #gen_02()
    #gen_03()
    #gen_04()
    #gen_05()
    gen_06()
    
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('comproot', nargs='+', help='mcstas-comps dirname searched recursively.')
    #parser.add_argument('ofile', help='Output file name')
    args = parser.parse_args()

    main(args)


