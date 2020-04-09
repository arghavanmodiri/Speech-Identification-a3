import os
import numpy as np
import os, fnmatch
import sys
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein istance.

    Works only for iterables up to 254 elements uint8).
    O(nm) time ans space complexity. 

    Parameters 
    ---------- 
    r : list of strings
    h : list of strings
    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    r_len = len(r)
    h_len = len(h)
    if r_len == 0:
        WER = np.inf
        return WER, 0, h_len, 0

    R = np.empty((r_len+1,h_len+1), int)
    B = np.empty((r_len+1,h_len+1), dtype="<U10")
    R[0,0] = 0
    for i in range(r_len+1):
        R[i,0] = max(i,0)
        B[i,0] = 'del'
    for j in range(h_len+1):
        R[0,j] = max(0,j)
        B[0,j] = 'ins'

    for i in range(1, r_len+1):
        for j in range(1, h_len+1):
            deletions = R[i-1, j] + 1
            if r[i-1] == h[j-1]:
                substitutions = R[i-1, j-1]
            else:
                substitutions = R[i-1, j-1]+1
            insertions = R[i,j-1]+1


            #R[i, j] = min(deletions, substitutions, insertions)
            # or below
            if substitutions <= deletions and substitutions <=insertions:
                R[i,j] = substitutions
                if r[i-1] == h[j-1]:
                    B[i,j] = 'ok'
                else:
                    B[i,j] = 'subs'
            elif insertions <= deletions and insertions < substitutions:
                R[i,j] = insertions
                B[i,j] = 'ins'
            else:
                R[i,j] = deletions
                B[i,j] = 'del'

    subs, ins, dels = backTrace(B, r_len, h_len)
    WER = 100 * R[r_len, h_len] / r_len
    return WER, subs, ins, dels

def backTrace(B_matrix, n, m):
    i = n
    j = m
    ins = 0
    subs = 0
    dels = 0
    while i > 0 or j > 0:
        temp = B_matrix[i, j]
        if temp == 'del':
            dels += 1
            i -= 1
        elif temp == 'ins':
            ins += 1
            j -= 1
        elif temp == 'subs':
            subs += 1
            j -= 1
            i -= 1
        elif temp == 'ok':
            j -= 1
            i -= 1

    return (subs, ins, dels)

def preproc(text):
    clean_text = text.split(" ", 2)[2].strip()
    clean_text = text.lower()
    clean_text = re.sub(r'[!"#\$%&\'\(\)\*\+,\-\./:;<=>\?@\\\^_`{\|}~/]', "", clean_text)
    return clean_text

if __name__ == "__main__":

    print(Levenshtein("who is there".split(), "is there".split()))
    print(Levenshtein("who is there".split(), "".split()))
    print(Levenshtein("".split(), "who is there".split()))
    print(Levenshtein("how to recognize speech".split(), "how to wreck a nice beach".split()))
    kald_WERs = []
    goog_WERs = []
    sys.stdout = open("asrDiscussion.txt", "w")
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*txt")

            #Read Files
            for file in files:
                with open(os.path.join(dataDir, speaker, file), 'r') as f:
                    if file == "transcripts.txt":
                        tran = f.readlines()
                    elif file == "transcripts.Google.txt":
                        goog = f.readlines()
                    elif file == "transcripts.Kaldi.txt":
                        kald = f.readlines()

            if len(tran) == 0 or len(goog) == 0 or len(kald) == 0:
                continue

            for idx in range(len(tran)):
                tran_clean = preproc(tran[idx])
                goog_clean = preproc(goog[idx])
                kald_clean = preproc(kald[idx])
                goog_score = Levenshtein(tran_clean.split(), goog_clean.split())
                kald_score = Levenshtein(tran_clean.split(), kald_clean.split())
                print("{} {} {} {} S:{}, I:{}, D:{}".format(speaker,
                                                            'Google',
                                                            idx,
                                                            goog_score[0],
                                                            goog_score[1],
                                                            goog_score[2],
                                                            goog_score[3]))

                print("{} {} {} {} S:{}, I:{}, D:{}".format(speaker,
                                                            'Kaldi',
                                                            idx,
                                                            kald_score[0],
                                                            kald_score[1],
                                                            kald_score[2],
                                                            kald_score[3]))
                kald_WERs.append(kald_score[0])
                goog_WERs.append(goog_score[0])

    kald_WERs = np.asarray(kald_WERs)
    goog_WERs = np.asarray(goog_WERs)
    print("Kaldi    mean:{}, std:{}".format(np.mean(kald_WERs), np.std(kald_WERs)))
    print("Google   mean:{}, std:{}".format(np.mean(goog_WERs), np.std(goog_WERs)))

