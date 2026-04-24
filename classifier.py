"""
Core logic for deciding whether a prior radiology study is relevant
to the current one. Works purely off study descriptions and dates.
"""

# Body-part groups.
# Keywords are matched as substrings of the *uppercased*, space-padded
# description so accidental partial hits are minimised.
BODY_GROUPS: dict[str, list[str]] = {
    "breast": [
        "BREAST", " MAM", "MAMMO", "MAMMOGR", "MAMMOGRAPHY",
        "DIGITAL SCREENER", "DIGITAL SCREEN", "STANDARD SCREENING COMBO",
        "SCREENER W CAD",
        # Bilateral / targeted breast ultrasounds often lack the word "breast"
        "ULTRASOUND BILAT SCREEN", "ULTRASOUND BILAT DIAG",
        "ULTRASOUND LT DIAG TARGET", "ULTRASOUND RT DIAG TARGET",
        "US BREAST", "US BI BREAST",
        " TOMO",
    ],
    "cardiac": [
        "CARDIAC", "CORONARY", "MYOCARDIAL", " MYO ", "NMMYO",
        " ECHO", " TTE", " TEE",
        "PERICARDIAL", "ECHOCARDIOG", "DOBUTAMINE",
    ],
    "chest": [
        "CHEST", " LUNG", "PULMON", "THORAX", "PLEURAL", "MEDIASTIN",
        " RIB", " V/Q", "ESOPHAG", "ESOPHAGRAM",
    ],
    "brain": [
        "BRAIN", " HEAD", "CRANIAL", "SKULL", " ORBIT", "SELLAR",
        "PITUITARY", "CEREBR", "INTRACRANIAL", "STROKE", "PERFUSION",
    ],
    "neck": [
        "NECK", "THYROID", "PARATHYROID", "SALIVARY", "PAROTID",
        "LARYNX", "PHARYNX", "SOFT TISSUE NECK",
    ],
    "spine_c": [
        "CERVICAL SPINE", "CERV SPINE", "C-SPINE", "C SPINE",
        "CERVICAL CORD", " CERVICAL",
    ],
    "spine_t": [
        "THORACIC SPINE", "T-SPINE", "T SPINE", "THORACIC CORD",
    ],
    "spine_l": [
        "LUMBAR", "LUMBOSACRAL", "L-SPINE", "L SPINE",
    ],
    "spine_s": [
        "SACRUM", "SACRAL", "COCCYX",
    ],
    "abdomen": [
        "ABDOM", " ABD/", " ABD ", "ABD^", "ABDOMEN^",
        " LIVER", "HEPATIC", "PANCREA", "SPLEEN",
        "GALLBLADDER", "BILIARY", " BOWEL", "COLON", "RECTUM",
        "GASTRIC", " GI SERIES", "CHOLANGIOGR",
    ],
    "pelvis": [
        "PELVIS", "PELVIC", "UTERUS", "UTERINE", "OVARY", "OVARIAN",
        "ENDOMETRI", " CERVIX", "VAGIN",
    ],
    "scrotal": [
        "SCROTUM", "SCROTAL", "TESTICULAR", "TESTIS", "TESTICLE",
    ],
    "prostate": ["PROSTATE"],
    "kidney": [
        "KIDNEY", " RENAL", "NEPHRO", " REN ", "CT PLV", "RENAL COLIC",
    ],
    "vascular_leg": [
        "VENOUS DOPPLER", "VENOUS IMAGING", "ARTERIAL IMAGING",
        "ARTERIAL STUDY", " ABI ", "CLAUDICATION",
    ],
    "vascular_head": [
        "TRANSCRANIAL DOPPLER",
    ],
    "carotid": [
        "CAROTID", "CT ANGIO CAROTID", "MRI ANGIO CAROTID",
        "CT ANGIOGRAM, CAROTID",
    ],
    "shoulder": ["SHOULDER"],
    "elbow":    ["ELBOW"],
    "wrist":    ["WRIST"],
    "hand":     [" HAND", "FINGER", "THUMB"],
    "hip":      [" HIP ", "HIP,", " FEMUR", "FEMORAL"],
    "knee":     ["KNEE", "PATELL"],
    "ankle":    ["ANKLE", "ACHILLES"],
    "foot":     [" FOOT", " FEET", " TOE ", "CALCANEUS", "HALLUX"],
    "leg":      ["TIBIA", "FIBULA", " LE "],
    # PET/CT skull-to-thigh is whole-body and relevant to most cross-sectional
    # imaging. Plain X-rays and routine echo typically don't cross-reference it.
    "wholebod": [
        "SKULL TO THIGH", "SKULLTHIGH", "SKULL-TO-THIGH",
        "WHOLE BODY PET", "TOTAL BODY PET",
        " PET/CT ", " PET-CT ",
    ],
    "bone_scan":    ["BONE SCAN", "NM BONE"],
    "bone_density": ["BONE DENSITY", " DEXA", " DXA"],
    "nuclear":  ["NUCLEAR MED", " SPECT "],
}

# Cross-relevance pairs – only kept when precision > 55% on the public split.
# Removed: cardiac↔chest (22%), carotid↔brain (33%), spine_c↔neck (24%).
CROSS_PAIRS: list[frozenset] = [
    frozenset({"abdomen", "pelvis"}),    # 56 % precision
    frozenset({"kidney", "abdomen"}),    # 69 %
    frozenset({"kidney", "pelvis"}),     # 69 %
    frozenset({"carotid", "neck"}),      # 67 %
    # Bone scans are used in oncology staging alongside cross-sectional imaging.
    frozenset({"bone_scan", "abdomen"}), # 75 %
    frozenset({"bone_scan", "pelvis"}),  # 71 %
    frozenset({"bone_scan", "chest"}),   # 60 %
]

CROSS_SET = {frozenset(p) for p in CROSS_PAIRS}

# Whole-body PET/CT is relevant to any cross-sectional (CT / MRI) study but
# NOT to plain X-rays, routine echo, or small-extremity studies.
PETCT_RELEVANT_GROUPS = {
    "chest", "brain", "abdomen", "pelvis", "kidney", "prostate",
    "neck", "spine_c", "spine_t", "spine_l", "breast", "cardiac",
}

# Groups where a laterality mismatch (RT vs LT) makes the prior irrelevant.
UNILATERAL_GROUPS = {
    "breast", "shoulder", "elbow", "wrist", "hand",
    "hip", "knee", "ankle", "foot", "leg",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_groups(desc: str) -> set[str]:
    d = " " + desc.upper() + " "
    found: set[str] = set()
    for grp, keywords in BODY_GROUPS.items():
        for kw in keywords:
            if kw in d:
                found.add(grp)
                break
    return found


def _get_laterality(desc: str) -> str:
    """Return 'right', 'left', 'both', or 'none'."""
    d = " " + desc.upper() + " "
    right_m = [" RT ", " RT,", " RT.", "-RT", " RIGHT ", " RUQ ", " RLQ "]
    left_m  = [" LT ", " LT,", " LT.", "-LT", " LEFT ",  " LUQ ", " LLQ "]
    both_m  = [" BILAT", " BILATERAL", " BILATERALLY", " BI ", " BOTH "]

    has_r = any(m in d for m in right_m)
    has_l = any(m in d for m in left_m)
    has_b = any(m in d for m in both_m)

    if has_b:
        return "both"
    if has_r and has_l:
        return "both"
    if has_r:
        return "right"
    if has_l:
        return "left"
    return "none"


def _groups_compatible(cg: set[str], pg: set[str]) -> bool:
    if cg & pg:
        return True
    for a in cg:
        for b in pg:
            if frozenset({a, b}) in CROSS_SET:
                return True
    return False


def _laterality_ok(cur_desc: str, pri_desc: str, shared: set[str]) -> bool:
    """False when laterality explicitly conflicts for unilateral body parts."""
    if not (shared & UNILATERAL_GROUPS):
        return True
    cur_lat = _get_laterality(cur_desc)
    pri_lat = _get_laterality(pri_desc)
    if cur_lat in ("none", "both") or pri_lat in ("none", "both"):
        return True
    return cur_lat == pri_lat


def _word_overlap(cur_desc: str, pri_desc: str) -> bool:
    """Simple word-overlap fallback when body-part detection misses both sides."""
    noise = {
        "CT", "MRI", "US", "XR", "WO", "WITH", "WITHOUT", "CON", "CONTRAST",
        "W", "W/O", "WO/W", "AND", "OR", "-", "LEFT", "RIGHT", "LT", "RT",
        "BI", "MIN", "VIEWS", "VIEW", "SCREEN", "SCREENING", "DX", "DIAG",
        "DIAGNOSTIC", "COMPLETE", "LIMITED", "BILATERAL", "BILAT", "AP", "PA",
        "LAT", "1", "2", "3", "4", "5", "GUIDED", "BIOPSY", "ADD", "ADDL",
    }
    cur_w = set(cur_desc.upper().split()) - noise
    pri_w = set(pri_desc.upper().split()) - noise
    return bool(cur_w & pri_w)


# ── Public API ───────────────────────────────────────────────────────────────

def is_relevant(current_desc: str, prior_desc: str) -> bool:
    """
    Returns True if the prior study should be shown alongside the current one.
    """
    cg = _get_groups(current_desc)
    pg = _get_groups(prior_desc)

    # Whole-body PET/CT is relevant to cross-sectional imaging of any covered region.
    if "wholebod" in pg and (cg & PETCT_RELEVANT_GROUPS):
        return True
    if "wholebod" in cg and (pg & PETCT_RELEVANT_GROUPS):
        return True

    # If body-part detection fails on one or both sides, fall back to text overlap.
    if not cg or not pg:
        return _word_overlap(current_desc, prior_desc)

    if not _groups_compatible(cg, pg):
        return False

    # For unilateral structures, a laterality conflict makes the prior irrelevant.
    shared = cg & pg
    if not _laterality_ok(current_desc, prior_desc, shared):
        return False

    return True
