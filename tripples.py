X = ["thinkers_begin, thinkers_midpoint",
     "thinkers_midpoint, thinkers_end"]
Y = ["tokens_begin, tokens_midpoint",
     "tokens_midpoint, tokens_end"]
Z = ["rows_begin, rows_midpoint",
     "rows_midpoint, rows_end"]

for x in X:
  for y in Y:
    for z in Z:
      print("        pack(%s," % (x))
      print("             %s," % (y))
      print("             %s);" % (z))
