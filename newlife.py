MAX_M = 5
MAX_N = 5
print()
print("switch ((std::min(m - m0, %d) << 4) | std::min(n - n0, %d)) {" % (MAX_M, MAX_N))
for VECTOR_REGISTERS in (32, 16):

  print("#if VECTOR_REGISTERS == %d" % (VECTOR_REGISTERS))

  a = []
  specified = set()
  for mc in range(1, MAX_M + 1):
    # if mc > 4 and mc % 4 != 0:
    #   continue
    for nc in range(1, MAX_N + 1):
      v = min(mc * nc + mc + 1,
              mc * nc + nc + 1)
      if v > VECTOR_REGISTERS:
        continue
      s = ""
      s += "case 0x%x%x:\n" % (mc, nc)
      s += "    mc = %d;\n" % (mc)
      s += "    nc = %d;\n" % (nc)
      s += "    gemm<%d, %d>(m0, m, n0, n);\n" % (mc, nc)
      s += "    break;"
      if mc % 8 == 0:
        v += 2
      if mc % 4 == 0:
        v += 1
      a.append((v, mc, nc, s, []))
      specified.add((mc, nc))

  a = list(reversed(sorted(a)))

  for mc in range(1, MAX_M + 1):
    for nc in range(1, MAX_N + 1):
      if (mc, nc) in specified:
        continue
      for v_, mc_, nc_, s_, extra in a:
        if mc_ <= mc and nc_ <= nc:
          extra.append("case 0x%x%x:" % (mc, nc))
          break

  for v, mc, nc, s, extra in a:
    for e in list(reversed(sorted(extra))):
      print(e)
    print(s)

  print("#endif")

print("default:")
print("    return;")
print("}")
