# -*- coding: utf-8 -*-

import getopt
import os
import random
import re
import sys
import math, codecs, string, fnmatch, unicodedata

from pan2020_author_veri import *


print ("UniNE_Verification .py loaded")

random.seed(1811)  # make results reproducible


if __name__ == '__main__':
    print ("UniNE.py started")

    inputFolder = ""
    outputFolder = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:")
    except getopt.GetoptError:
        print ("UniNE.py -i <inFolder> -o <outFolder>")  # Tira command format
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            inputFolder = arg
        elif opt == "-o":
            outputFolder = arg

    assert len(inputFolder) > 0      # if not true, stop the process
    print ("Input folder is", inputFolder)
    assert len(outputFolder) > 0
    print ("Output folder is", outputFolder)
#
#  Start here the real procedure
#
    Authorship_Verification (inputFolder, outputFolder)

    exit(0)
