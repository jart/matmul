MAX_THINKERS = 1
MAX_TOKENS = 5
MAX_ROWS = 5

bits = max(MAX_THINKERS.bit_length(),
           MAX_TOKENS.bit_length(),
           MAX_ROWS.bit_length())

print()

print("switch (std::min(thinkers_end - thinkers_begin, %d) << %d |" % (MAX_THINKERS, bits*2))
print("        std::min(tokens_end - tokens_begin, %d) << %d |" % (MAX_TOKENS, bits))
print("        std::min(rows_end - rows_begin, %d)) {" % (MAX_ROWS))
for VECTOR_REGISTERS in (32, 16):

  print("#if VECTOR_REGISTERS == %d" % (VECTOR_REGISTERS))

  a = []
  specified = set()
  for thn in range(1, MAX_THINKERS + 1):
    for ton in range(1, MAX_TOKENS + 1):
      for ron in range(1, MAX_ROWS + 1):
        v = min(thn*ton*ron + thn + ton,
                thn*ton*ron + thn + ron)
        if v > VECTOR_REGISTERS:
          continue
        s = ""
        s += "case %d<<%d | %d<<%d | %d:\n" % (thn, bits*2, ton, bits, ron)
        s += "    thn = %d;\n" % (thn)
        s += "    ton = %d;\n" % (ton)
        s += "    ron = %d;\n" % (ron)
        s += "    kernel<%d, %d, %d>(thinkers_begin, thinkers_end, tokens_begin, tokens_end, rows_begin, rows_end);\n" % (thn, ton, ron)
        s += "    break;"
        v -= abs(thn - ton)
        v -= abs(thn - ron)
        v -= abs(ron - ton)
        v += (ron % 2 == 0) * (ron / 2)
        a.append((v, thn, ton, ron, s, []))
        specified.add((thn, ton, ron))

  a = list(reversed(sorted(a)))

  for thn in range(1, MAX_THINKERS + 1):
    for ton in range(1, MAX_TOKENS + 1):
      for ron in range(1, MAX_ROWS + 1):
        if (thn, ton, ron) in specified:
          continue
        for v_, thn_, ton_, ron_, s_, extra in a:
          if thn_ <= thn and ton_ <= ton and ron_ <= ron:
            extra.append("case %d<<%d | %d<<%d | %d:" % (thn, bits*2, ton, bits, ron))
            break

  for v, thn, ton, ron, s, extra in a:
    for e in list(reversed(sorted(extra))):
      print(e)
    print(s)

  print("#endif")

print("default:")
print("    return;")
print("}")
