def ismember_custom(A, B):
    Bs, sortInds = sorted(B), sorted(range(len(B)), key=lambda x: B[x])
    _, firstInds = _ismemberhelper(A, Bs)
    lastInds = ismembc2(A, Bs)
    allInds = [sortInds[x:y-(x==0)] for x, y in zip(firstInds[:, 0], lastInds[:, 0])]
    cellsz = [len(x) > 1 for x in allInds]
    temp = [cellsz[i] == 1 for i in range(len(cellsz))]
    if any(temp):
        for i in range(len(temp)):
            if temp[i] == 1:
                allInds[i] = [x for x in allInds[i] if ismembc2(A[i, 2], B[x, 2])]
    index = [item for sublist in allInds for item in sublist]
    return index