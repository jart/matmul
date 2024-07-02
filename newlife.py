MAX_M = 5
MAX_N = 5
print()
for VECTOR_REGISTERS in (32, 16):

  print("#if VECTOR_REGISTERS == %d" % (VECTOR_REGISTERS))

  # choose tile size that exploits all vector registers
  specified = {}
  for mc in range(1, MAX_M + 1):
    for nc in range(1, MAX_N + 1):
      memory_loads = mc + nc
      accumulators = mc * nc
      v = accumulators + memory_loads // 2
      if v <= VECTOR_REGISTERS:
        if mc % 8 == 0:
          v += 2
        if mc % 4 == 0:
          v += 1
        specified[mc, nc] = v

  # generate code for handling biggest tile (e.g. 5x5)
  # generate code for handling edge tiles (i.e. <=2x2)
  # avoid long compile times to generate tiles between
  (best_mc, best_nc), best_v = list(sorted(specified.items(), key=lambda s: s[1]))[-1]
  for (mc, nc), v in list(specified.items()):
    if v < best_v and (mc > 2 or nc > 2):
      del specified[mc, nc]

  print("switch ((std::min(m - m0, %d) << 4) | std::min(n - n0, %d)) {" % (best_mc, best_nc))

  a = []
  for (mc, nc), v in specified.items():
    s = ""
    s += "case 0x%x%x:\n" % (mc, nc)
    s += "    mc = %d;\n" % (mc)
    s += "    nc = %d;\n" % (nc)
    s += "    gemm<%d, %d>(m0, m, n0, n);\n" % (mc, nc)
    s += "    break;"
    a.append((v, mc, nc, s, []))

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

  print("default:")
  print("    return;")
  print("}")
  print("#endif")
