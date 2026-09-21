# -*- coding: utf-8 -*-
"""Tu dien da ngon ngu cho bo rut nhan yeu.

Bo cuc mot nhan gom hai phan tach roi - day la quyet dinh thiet ke quan trong nhat cua
ca module:

  * `concept` - cau truc giai phau hoac khoang duoc nhac toi (vd: "menisco interno").
  * `finding` - bang chung benh ly (vd: "rotura", "tear").

Ly do tach: nhac ten cau truc KHONG dong nghia voi chan doan duong tinh. Trong cac bao cao
tieng Anh co nhan nguoi gan, tu "fracture" xuat hien o ca ca duong tinh lan am tinh; cum tu
ACL va effusion xuat hien o gan nhu moi bao cao bat ke nhan. Thong tin chan doan nam o tu
xung quanh - "torn", "intact", "no" - chu khong o ten cau truc.

Hai kieu nhan:
  * TYPE_AB ("concept + finding"): chan thuong day chang / menisci / thoai hoa khop. Phai co
    ca hai moi tinh la duong tinh.
  * TYPE_B ("finding la chinh no"): effusion, synovitis, nang Baker, dap xuong, gay xuong.
    Ban than tu da la bat thuong, chi can khong bi phu dinh.

Chin ngon ngu co mat trong du lieu: Anh, Tay Ban Nha, Tho Nhi Ky, Croatia, Hy Lap, Duc,
Bulgaria, Ha Lan, Phap. Tu dien khong the phu het bien the - phan khong khop duoc tra ve
`unknown`, KHONG tra ve 0.
"""
from __future__ import annotations

TYPE_AB = "concept_plus_finding"
TYPE_B = "finding_only"

# --- Tu phu dinh, gom chung moi ngon ngu -------------------------------------------------
# Dat truoc dau hieu benh ly thi lat nguoc ket luan. Vi du that trong du lieu:
#   "Medial meniscus is not torn." / "sin signos de rotura" / "No fracture is seen."
NEGATION_CUES = [
    # en
    r"\bno\b", r"\bnot\b", r"\bnor\b", r"\bwithout\b", r"\babsent\b", r"\bfree of\b",
    r"\bnegative for\b", r"\bno evidence\b", r"\bruled out\b",
    # es
    r"\bsin\b", r"\bausencia\b", r"\bausente\b", r"\bdescarta",
    # de
    r"\bkein\w*\b", r"\bohne\b", r"\bnicht\b",
    # fr
    r"\bpas de\b", r"\bsans\b", r"\babsence\b", r"\bpas d'",
    # nl
    r"\bgeen\b", r"\bzonder\b",
    # tr (hau to phu dinh cua dong tu: izlenmemistir / saptanmamistir / gorulmemistir)
    r"\byok\b", r"\bmemi[ss]tir\b", r"\bmami[ss]tir\b", r"\bsaptanma", r"\bizlenme",
    # hr
    r"\bnema\b", r"\bbez\b",
    # el
    r"δεν\b", r"χωρ[ίι]ς\b", r"ουδ[εέ]ν",
    # bg
    r"не\b", r"без\b", r"липсва",
]

# --- Khang dinh binh thuong -------------------------------------------------------------
# Khac phu dinh: khong phu dinh mot tu benh ly nao ca, ma khang dinh truc tiep la binh thuong.
# "ACL is intact." khong chua tu phu dinh nao nhung la bang chung am tinh ro rang.
NORMAL_CUES = [
    r"\bintact\b", r"\bpreserved\b", r"\bnormal\w*\b", r"\bunremarkable\b",
    r"\bint[ae]gr\w*\b", r"\bconservad\w*\b", r"l[íi]mites normales", r"\bhabitual\b",
    r"\bregelrecht\b", r"\bintakt\b", r"\bunauff\w*llig\b", r"\bunverse\w*hrt\b",
    r"\brespect[ée]\w*\b",
    r"\bdo[ğg]al\b", r"\bsa[ğg]lam\b", r"\bnormald[ıi]r\b",
    r"\buredan\b", r"\buredno\b", r"\bintaktan\b",
    r"φυσιολογικ", r"ακ[εέ]ραι",
    r"нормал", r"интакт",
]

# --- Dau hieu benh ly dung cho nhom day chang / menisci ---------------------------------
TEAR_FINDINGS = [
    r"\btear\w*\b", r"\btorn\b", r"\bruptur\w*\b", r"\bdisrupt\w*\b", r"\bdiscontinu\w*\b",
    r"\bavuls\w*\b", r"\bsprain\w*\b", r"\bfray\w*\b", r"\blesion\w*\b", r"\binjur\w*\b",
    r"\bpartial thickness\b", r"\bfull thickness\b", r"\bmacerat\w*\b", r"\bextrusion\b",
    r"\brotur\w*\b", r"\bdesgarr\w*\b", r"\blesi[óo]n\w*\b",
    r"\briss\b", r"\beinriss\w*\b", r"\bl[äa]sion\w*\b",
    r"\bd[ée]chirure\w*\b", r"\bfissur\w*\b", r"\bl[ée]sion\w*\b",
    r"\bscheur\w*\b", r"\bletsel\b", r"\bruptuur\b",
    # Tieng Tho bien am k -> g/ğ khi them hau to: yirtik -> yirtigi, yirtiklari.
    # Thieu bien the nay la bo lot ca nhom 546 bao cao tieng Tho (12.4% dataset).
    r"\by[ıi]rt[ıi][kğg]\w*\b", r"\br[üu]pt[üu]r\w*\b", r"\blezyon\w*\b", r"\byaralanma\w*\b",
    r"\bpuknu\w*\b", r"\blezij\w*\b", r"\bo[šs]te[ćc]enj\w*\b",
    r"ρ[ήη]ξ", r"βλ[άα]β",
    r"разкъсван", r"руптур", r"лези",
]

# Dau hieu thoai hoa khop dung chung cho ba nhan OA.
OA_FINDINGS = [
    r"\bosteoarthr\w*\b", r"\bosteoarthros\w*\b", r"\bgonarthros\w*\b", r"\barthros\w*\b",
    r"\bchondropath\w*\b", r"\bchondromalac\w*\b", r"\bosteophyt\w*\b",
    r"\bcartilage (?:loss|thinning|defect|wear|fissur\w*|degenerat\w*|heterogene\w*)\b",
    r"\bchondral (?:loss|thinning|defect|wear|fissur\w*)\b", r"\bosteochondral defect\b",
    r"\bjoint space (?:narrow\w*|loss)\b", r"\bsubchondral (?:cyst|sclerosis)\w*\b",
    r"\bartrosis\b", r"\bcondropat[íi]a\b", r"\bosteofit\w*\b", r"[úu]lcera condral",
    r"p[ée]rdida de cart[íi]lago", r"\bcondral\w*\b",
    r"\barthrose\b", r"\bgonarthrose\b", r"\bknorpel\w*\b",
    r"\bchondropathie\b", r"\bost[ée]ophyt\w*\b", r"\bchondrop\w*\b",
    r"\bartrose\b", r"\bkraakbeen\w*\b",
    r"\bosteoartr\w*\b", r"\bk[ıi]k[ıi]rdak\b", r"\bkondropat\w*\b",
    r"\bartroz\w*\b", r"\bhrskavic\w*\b",
    r"αρθρ[ίι]τιδ", r"οστε[όο]φυτ", r"χ[όο]νδρ",
    r"артроз", r"остеофит", r"хрущял",
]

# Tu chi muc do nhe - dung cho cau hinh `mild_oa_negative` o ngay 8.
MILD_CUES = [
    r"\bmild\w*\b", r"\bminimal\w*\b", r"\bslight\w*\b", r"\btrace\b", r"\bearly\b",
    r"\bleve\b", r"\bm[íi]nim\w*\b", r"\bdiscret\w*\b", r"\bgering\w*\b", r"\bleicht\w*\b",
    r"\bl[ée]g[èe]r\w*\b", r"\bminime\b", r"\bhafif\w*\b", r"\bblag\w*\b",
    r"[ήη]πι", r"лек",
]

# --- Tu dien theo tung nhan --------------------------------------------------------------
LEXICON: dict[str, dict] = {
    "ACL": {
        "type": TYPE_AB,
        "concept": [
            r"\bACL\b", r"\banterior cruciate\b", r"cruzado anterior", r"\bLCA\b",
            r"vorder\w* kreuzband", r"\bVKB\b", r"crois[ée]\w* ant[ée]rieur",
            r"voorste kruisband", r"[öo]n [çc]apraz", r"\b[ÖO][ÇC]B\b",
            r"prednj\w* ukri[žz]en\w*",
            r"πρ[όο]σθι\w* χιαστ",
            r"предна кръстна",
        ],
        "finding": TEAR_FINDINGS,
    },
    "MCL": {
        "type": TYPE_AB,
        "concept": [
            r"\bMCL\b", r"\bmedial collateral\b", r"\btibial collateral\b",
            r"colateral (?:medial|interno)", r"\bLCM\b", r"\bLLI\b",
            r"\binnenband\b", r"medial\w* kollateralband",
            r"collat[ée]ral\w* (?:m[ée]dial|interne)",
            r"mediale collaterale band", r"i[çc] yan ba[ğg]",
            r"medijaln\w* kolateraln\w*",
            r"[έε]σω πλ[άα]γι",
            r"медиален колатерал",
        ],
        "finding": TEAR_FINDINGS,
    },
    "Medial Meniscus": {
        "type": TYPE_AB,
        "concept": [
            r"\bmedial meniscus\b", r"menisco (?:medial|interno)",
            r"\binnenmeniskus\b", r"medial\w* meniskus",
            r"m[ée]nisque (?:m[ée]dial|interne)", r"mediale meniscus",
            r"(?:i[çc]|medial) men[ıi]sk[üu]s", r"medijaln\w* meniskus",
            r"[έε]σω μην[ίι]σκ",
            r"медиален мениск",
        ],
        "finding": TEAR_FINDINGS,
    },
    "Lateral Meniscus": {
        "type": TYPE_AB,
        "concept": [
            r"\blateral meniscus\b", r"menisco (?:lateral|externo)",
            r"au[ßs]enmeniskus", r"lateral\w* meniskus",
            r"m[ée]nisque (?:lat[ée]ral|externe)", r"laterale meniscus",
            r"(?:d[ıi][şs]|lateral) men[ıi]sk[üu]s", r"lateraln\w* meniskus",
            r"[έε]ξω μην[ίι]σκ",
            r"латерален мениск",
        ],
        "finding": TEAR_FINDINGS,
    },
    "Medial OA": {
        "type": TYPE_AB,
        "concept": [
            r"\bmedial (?:compartment|femoral condyle|tibial plateau|femorotibial|patellar facet)\b",
            r"compartimento (?:medial|interno)", r"femorotibial\w* (?:medial|interno)",
            r"medial\w* (?:kompartiment|femurkondyl|gelenkspalt)",
            r"compartiment (?:m[ée]dial|interne)", r"mediale compartiment",
            r"medial kompartman", r"medijaln\w* (?:odjelj|kompartm)",
            r"[έε]σω (?:διαμ[έε]ρισμα|μηροκνημ)",
            r"медиалн\w* (?:отдел|компартм)",
        ],
        "finding": OA_FINDINGS,
    },
    "Lateral OA": {
        "type": TYPE_AB,
        "concept": [
            r"\blateral (?:compartment|femoral condyle|tibial plateau|femorotibial)\b",
            r"compartimento (?:lateral|externo)", r"femorotibial\w* (?:lateral|externo)",
            r"lateral\w* (?:kompartiment|femurkondyl|gelenkspalt)",
            r"compartiment (?:lat[ée]ral|externe)", r"laterale compartiment",
            r"lateral kompartman", r"lateraln\w* (?:odjelj|kompartm)",
            r"[έε]ξω (?:διαμ[έε]ρισμα|μηροκνημ)",
            r"латералн\w* (?:отдел|компартм)",
        ],
        "finding": OA_FINDINGS,
    },
    "PF OA": {
        "type": TYPE_AB,
        "concept": [
            r"\bpatellofemoral\b", r"\bpatellar facet\b", r"\btrochlea\w*\b", r"\bretropatellar\b",
            r"\bfemoropatelar\w*\b", r"\bpatelofemoral\w*\b", r"\br[óo]tul\w*\b", r"\btr[óo]clea\w*\b",
            r"\bfemoropatellar\w*\b", r"\bpatella\w*\b", r"\bkniescheibe\b",
            r"f[ée]moro[- ]?patellaire", r"\bpatellofemorale\b",
            r"patellofemoral kompartman", r"patelofemoraln\w*",
            r"επιγονατιδομηρια",
            r"пателофеморал",
        ],
        "finding": OA_FINDINGS,
    },
    "Effusion": {
        "type": TYPE_B,
        "concept": [
            r"\beffusion\w*\b", r"\bjoint fluid\b", r"fluid (?:accumulat|collect)\w*",
            r"derrame articular", r"l[íi]quido articular",
            r"\bgelenkerguss\b", r"\bergu[ßs]\w*\b",
            r"\b[ée]panchement\w*\b", r"\bgewrichtsvocht\b", r"\beffusie\b",
            r"\bef[üu]zyon\w*\b", r"eklem s[ıi]v[ıi]s[ıi]", r"\bizljev\w*\b",
            r"(?:αρθρικ|ένδαρθρ)\w* (?:υγρ|συλλογ)",
            r"излив",
        ],
        "finding": None,
    },
    "Synovitis": {
        "type": TYPE_B,
        "concept": [
            r"\bsynovit\w*\b", r"synovial (?:thicken|proliferat|hypertroph)\w*",
            r"thickened synovi\w*", r"\bsinovit\w*\b", r"\bsynovialitis\b",
            r"υμεν[ίι]τιδ", r"синовит",
        ],
        "finding": None,
    },
    "Baker's": {
        "type": TYPE_B,
        "concept": [
            r"baker'?s? cyst\w*", r"popliteal cyst\w*",
            r"quiste popl[íi]te\w*", r"\bbakerzyste\b", r"\bpoplitealzyste\b",
            r"kyste popl[éi]t\w*", r"\bbakercyste\b", r"popliteale cyste",
            r"baker kist\w*", r"popliteal kist\w*", r"bakerova cist\w*",
            r"κ[ύυ]στη\w* (?:baker|ιγνυακ)",
            r"киста на бейкър",
        ],
        "finding": None,
    },
    "Contusion": {
        "type": TYPE_B,
        "concept": [
            r"\bcontusion\w*\b", r"bone bruis\w*",
            r"(?:bone )?marrow o?edema", r"subchondral o?edema", r"\bbone marrow o?edema\b",
            r"contusi[óo]n\w*", r"edema [óo]se\w*", r"edema de m[ée]dula",
            r"knochenmark[öo]dem", r"\bkontusion\w*\b",
            r"contusion osseuse", r"[œoe]d[èe]me (?:osseux|m[ée]dullaire)",
            r"\bbotcontusie\b", r"beenmerg[oe]+deem",
            r"kemik (?:ili[ğg]i )?[öo]dem\w*", r"kont[üu]zyon\w*",
            r"\bkontuzij\w*\b", r"nagnje[čc]enj\w*",
            r"ο[ίι]δημα μυελο",
            r"костн\w* оток", r"контузи",
        ],
        "finding": None,
    },
    "Fracture": {
        "type": TYPE_B,
        "concept": [
            r"\bfractur\w*\b", r"\bfraktur\w*\b", r"\bfractuur\b",
            r"\bk[ıi]r[ıi]k\w*\b", r"\bprijelom\w*\b",
            r"κ[άα]ταγμα", r"фрактур", r"счупван",
        ],
        "finding": None,
    },
}
