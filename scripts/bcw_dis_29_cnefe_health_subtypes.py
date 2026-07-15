# %% [markdown]
# -------------------------------------------------------------------------
## bcw_dis_29_cnefe_health_subtypes — what is inside CNEFE "saúde" (species 5)?
# -------------------------------------------------------------------------
"""
Diagnostic: the CNEFE healthcare count (species 5) is ~5.7x the OSM healthcare
count. This checks whether CNEFE bundles in establishment sub-types that the OSM
filter does not capture, by mining the free-text name field DSC_ESTABELECIMENTO
of every species-5 record in São Paulo municipality and bucketing it by keyword.
"""
# %%
import re
import unicodedata
import pandas as pd
import bcw_dis_00_config as cfg

SP_MUNI = "3550308"
CNEFE_CSV = cfg.RAW_DIR / "cnefe" / "35_SP.csv"
CHUNK = 2_000_000


def norm(s):
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", s).strip()


BUCKETS = [
    ("pharmacy",        ["FARMAC", "DROGA"]),
    ("dental",          ["ODONTO", "DENTAR", "DENTIST"]),
    ("hospital",        ["HOSPITAL", "PRONTO SOCORRO", "PRONTO-SOCORRO", "MATERNIDADE", "SANTA CASA"]),
    ("public_primary",  ["UBS", "UNIDADE BASICA", "POSTO DE SAUDE", "CENTRO DE SAUDE",
                          "AMA ", "UPA", "PSF", "ESTRATEGIA SAUDE", "AMBULATORIO"]),
    ("lab_diagnostic",  ["LABORAT", "DIAGNOSTIC", "ANALISES CLINIC", "IMAGEM", "RADIOLOG",
                          "ULTRASSOM", "TOMOGRAF", "RESSONANCIA", "VACINA"]),
    ("clinic",          ["CLINICA", "CLINIC", "POLICLINIC", "CENTRO MEDICO", "CENTRO CLINICO"]),
    ("office",          ["CONSULTORIO"]),
    ("therapy",         ["FISIOTERAP", "PSICOLOG", "FONOAUDIOL", "TERAPIA", "NUTRIC"]),
    ("optics",          ["OTICA", "OPTICA"]),
    ("veterinary",      ["VETERINAR", "PET SHOP", "PETSHOP"]),
]


def classify(name):
    for label, kws in BUCKETS:
        if any(k in name for k in kws):
            return label
    return "other/uncategorised" if name else "(blank name)"


print("Reading CNEFE species-5 (health) records ...", flush=True)
names = []
for i, ch in enumerate(pd.read_csv(
        CNEFE_CSV, sep=";", usecols=["COD_MUNICIPIO", "COD_ESPECIE", "DSC_ESTABELECIMENTO"],
        dtype=str, encoding="latin-1", chunksize=CHUNK)):
    ch = ch[(ch["COD_MUNICIPIO"] == SP_MUNI) & (ch["COD_ESPECIE"] == "5")]
    if not ch.empty:
        names.append(ch["DSC_ESTABELECIMENTO"])
    print(f"    chunk {i} done", flush=True)

s = pd.concat(names, ignore_index=True).map(norm)
total = len(s)
blank = (s == "").sum()
print(f"\nTotal CNEFE health (species 5) records in São Paulo: {total:,}")
print(f"  with a name: {total - blank:,}  | blank name: {blank:,}")

bucket = s.map(classify)
tab = bucket.value_counts()
print("\n=== CNEFE health records by keyword bucket ===")
for k, v in tab.items():
    print(f"  {k:22} {v:7,}  ({v/total:5.1%})")

print("\n=== top 30 raw establishment names (species 5) ===")
print(s[s != ""].value_counts().head(30).to_string())
