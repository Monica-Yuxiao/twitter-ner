from ner.helper import *

if contains_special("-Tim-"):
    print("-Tim-")

if contains_special("Keygen"):
    print("Keygen")


if contains_special("@Keygen"):
    print("@Keygen")


if contains_special("Keygen./"):
    print("Keygen./")

print(contains_alphabet("..."))