#!/usr/bin/env python

import string

SIZE = 64 # Data img size in px

LETTERS = 26 # Number of letters in samples

ALPHABET = list(string.ascii_uppercase) # A..Z

LETTERPOS = dict() # Position of each letter in samples
for i, letter in enumerate(ALPHABET):
    LETTERPOS[letter] = i*SIZE
