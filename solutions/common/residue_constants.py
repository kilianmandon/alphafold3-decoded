from atomworks.constants import STANDARD_AA, UNKNOWN_AA, STANDARD_RNA, UNKNOWN_RNA, UNKNOWN_DNA, STANDARD_DNA, GAP

AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    *STANDARD_AA, UNKNOWN_AA,
    # 1 gap
    GAP,
    # 4 RNA
    *STANDARD_RNA, UNKNOWN_RNA,
    # 4 DNA
    *STANDARD_DNA, UNKNOWN_DNA
)

AF3_TOKENS_MAP = dict(zip(AF3_TOKENS, range(len(AF3_TOKENS))))
AF3_TOKENS_MAP[UNKNOWN_RNA] = AF3_TOKENS_MAP[UNKNOWN_AA]
AF3_TOKENS_MAP[UNKNOWN_DNA] = AF3_TOKENS_MAP[UNKNOWN_AA]



PROTEIN_TO_ID = {
    'A': 0,
    'B': 3,  # Same as D.
    'C': 4,
    'D': 3,
    'E': 6,
    'F': 13,
    'G': 7,
    'H': 8,
    'I': 9,
    'J': 20,  # Same as unknown (X).
    'K': 11,
    'L': 10,
    'M': 12,
    'N': 2,
    'O': 20,  # Same as unknown (X).
    'P': 14,
    'Q': 5,
    'R': 1,
    'S': 15,
    'T': 16,
    'U': 4,  # Same as C.
    'V': 19,
    'W': 17,
    'X': 20,
    'Y': 18,
    'Z': 6,  # Same as E.
    '-': 21,
}