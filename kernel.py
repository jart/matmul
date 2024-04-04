RM = 2
RN = 2

print()
print("dontinline void gemm%dx%d(int m0, int m, int n0, int n) {" % (RM, RN))
print("    BEGIN_KERNEL(%d, %d)" % (RM, RN))
for i in range(RM):
  for j in range(RN):
    print("    V c%d%d = zero();" % (i, j))
print("    for (int l = 0; l < k; l += KN) {")
for i in range(RN):
  print("        V k%d = load(B + ldb * (j + %d) + l);" % (i, i))
for i in range(RM):
  print("        V a%d = load(A + lda * (i + %d) + l);" % (i, i))
  for j in range(RN):
    print("        c%d%d = madd(a%d, k%d, c%d%d);" % (i, j, i, j, i, j))
print("    }")
for j in range(RN):
  for i in range(RM):
    print("    C[ldc * (j + %d) + (i + %d)] = hsum(c%d%d);" % (j, i, i, j))
print("    END_KERNEL()")
print("}")
