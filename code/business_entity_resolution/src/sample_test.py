"""Smoke-test pre-process.py functions on tiny hand-made samples.

Full data is too heavy to run blindly (12.5M rows); this validates each
component against known tricky records instead.
"""
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import pre_process_shim as pp  # real module is pre-process.py (dashed filename)


def main():
    src1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Orelee's Barbershop", "Prime Money Pvt. Ltd.", ""],
        "business_address": ["1795 Westchester Drive, High Point, NC",
                             "17560 Ellis Road, Tahlequah, OK", "x"],
        "country": ["US", "US", "US"],
    })
    src2 = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ["\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921",
                          "-- Holloway Peak Inc Seafood"],
        "business_address": ["KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
                             "105 ELM ST, MORGANTON, NC"],
        "country": ["India", "US"],
    })
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1", "", "S1-2"],
        "matched_entity_ids": ["S2-1", "S2-2", ""],
    })

    print("== clean_source drops empty-name row ==")
    c1 = pp.clean_source(src1.copy(), "S1")
    assert len(c1) == 2, c1
    print(c1[["entity_id", "name_raw"]].to_string())
    print("note: S1-3 with empty name dropped")

    print("\n== ground truth never dropped, empties tokenized ==")
    g = pp.load_ground_truth_df(gt)
    assert len(g) == 3
    assert (g["source1_entity_id"] == "<empty>").sum() == 1
    assert (g["matched_entity_ids"] == "<empty>").sum() == 1
    print(g.to_string())

    print("\n== normalization ==")
    assert pp.normalize_name("Orelee's Barbershop") == "orelee s barbershop"
    assert pp.normalize_name("Prime Money Pvt. Ltd.") == "prime money"
    assert pp.normalize_name("-- Holloway Peak Inc Seafood") == "holloway peak seafood"
    # accented Latin folds, Devanagari survives intact
    cafe_pp = pp.normalize_name("Café Müller GmbH")
    assert cafe_pp == "cafe muller", cafe_pp
    hindi_raw = "\u0930\u093e\u092e \u092e\u093e\u0930\u094d\u0915\u0947\u091f\u093f\u0902\u0917"
    hindi = pp.normalize_name(hindi_raw)
    assert hindi != "" and "\u093e" in hindi, "Devanagari matras must survive"
    # UTF-8-as-Latin-1 mojibake punctuation is dropped, real accents stay
    assert pp.normalize_text("Sector Â\u0080\u0093 V") == "sector v"
    assert pp.normalize_text("Tvdâ\u0080\u0099s Colony") == "tvd s colony"
    assert pp.normalize_text("Château") == "chateau"
    # French legal forms, incl. dotted acronyms, are stripped
    for raw, exp in [("Ensemble & Fils SASU", "ensemble fils"),
                     ("Mountain Club S.A.R.L.", "mountain club"),
                     ("ZSJ Club E.U.R.L.", "zsj club"),
                     ("ETS POSTIERS SARL CIE", "ets postiers"),
                     ("Chasseurs (France) Services SCI", "chasseurs france services"),
                     ("Toute & Cie SAS", "toute"),
                     ("A.P Management Pvt. Ltd.", "ap management"),
                     ("Acme L.L.C.", "acme")]:
        got = pp.normalize_name(raw)
        assert got == exp, (raw, got, exp)
    # "private limited" / "pvt ltd" / "llp" in every Indic script seen in data
    for raw in ["राम प्राइवेट लिमिटेड", "राम प्रा. लि.",
                "राम एलएलपी", "ராம் பிரைவேட் லிமிடெட்", "ராம் எல்எல்பி",
                "రామ్ ప్రైవేట్ లిమిటెడ్", "రామ్ ఎల్‌ఎల్‌పీ",
                "ರಾಮ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "ರಾಮ್ ಎಲ್ಎಲ್‌ಪಿ",
                "রাম প্রাইভেট লিমিটেড", "রাম এলএলপি",
                "રામ પ્રાઇવેટ લિમિટેડ", "રામ પ્રા. લિ.", "રામ એલએલપી",
                "രാം പ്രൈവറ്റ് ലിമിറ്റഡ്", "രാം എൽഎൽപി",
                "ରାମ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", "ରାମ ଏଲ୍‌ଏଲ୍‌ପି",
                "ਰਾਮ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ", "ਰਾਮ ਪ੍ਰਾ. ਲਿ.", "ਰਾਮ ਐਲਐਲਪੀ"]:
        got = pp.normalize_name(raw)
        assert len(got.split()) == 1, (raw, got)
    print("  'Orelee's Barbershop' ->", repr(pp.normalize_name("Orelee's Barbershop")))
    print("  'Prime Money Pvt. Ltd.' ->", repr(pp.normalize_name("Prime Money Pvt. Ltd.")))
    print("  Hindi name ->", repr(hindi)[:40], "... (non-empty, Devanagari preserved)")

    print("\n== char n-grams ==")
    grams = pp.char_ngrams("abc de")
    assert grams[0] == " ab" and "abc" in grams and grams[-1] == "de ", grams
    short = pp.char_ngrams("ab cd")
    assert short == [" ab", "ab ", "b c", " cd", "cd "], short
    assert pp.char_ngrams("") == []
    print("  'abc de' ->", grams)
    print("  short 'ab cd' ->", short)

    print("\n== canopy keys ==")
    k1 = pp.canopy_keys("orelee s barbershop", "1795 westchester drive high point nc")
    k2 = pp.canopy_keys("holloway peak seafood", "105 elm st morganton nc")
    assert all(k.startswith(("N1", "N2", "N3", "N4", "N5", "N6", "N7", "A1")) for k in k1)
    assert pp.canopy_keys("", "") == []
    # two similar names should share at least one canopy key
    k3 = pp.canopy_keys("orelee s barber shop", "1795 westchester drive high point nc")
    assert set(k1) & set(k3), (k1[:3], k3[:3])
    print("  orelee: %d keys, e.g. %s" % (len(k1), k1[:3]))
    print("  holloway: %d keys, e.g. %s" % (len(k2), k2[:3]))
    print("  perturbed 'orelee s barber shop' shares canopy: OK")

    print("\n== phonetic keys ==")
    # same sound different spelling -> share at least one phonetic bucket
    # (metaphone OR soundex; e.g. Sharma/Sarma differ in metaphone but match
    # on soundex, Muller/Mullar match on metaphone)
    for a, b in [("muller", "mullar"), ("sharma", "sarma"),
                 ("krishna", "krishna  ")]:
        buckets_a = pp.phonetic_buckets(a)
        buckets_b = pp.phonetic_buckets(b)
        assert buckets_a and buckets_a & buckets_b, (a, b, buckets_a, buckets_b)
    # phonetic differs from raw text
    assert pp.phonetic_key("muller") == pp.phonetic_key("mular"), \
        "phonetic should be spelling-invariant"
    # variants include last-token metaphone / joint skeleton
    vs = pp.phonetic_variants("orelee barbershop")
    assert len(vs) >= 1, vs
    # empty name -> empty key, no crash
    assert pp.phonetic_key("") == ""
    assert pp.phonetic_variants("") == []
    # N7 canopy family: misspelled pair must share at least one canopy
    k_exact = pp.canopy_keys("muller trading co", "12 main st")
    k_typo = pp.canopy_keys("mular trading  co", "12 main st")
    exact_ascii = [k for k in k_exact if k.startswith("N7")]
    typo_ascii = [k for k in k_typo if k.startswith("N7")]
    assert set(exact_ascii) & set(typo_ascii), (exact_ascii, typo_ascii)
    # non-ascii (hindi) names: N7 skipped rather than crashing
    hindi_keys = pp.canopy_keys(pp.normalize_name(hindi_raw), "")
    assert all(not k.startswith("N7") for k in hindi_keys), hindi_keys[:2]
    print("  phonetic_key('muller') == phonetic_key('mular'):", pp.phonetic_key("muller"))
    print("  N7 canopy shared by 'muller trading co' and 'mular trading  co': OK")
    print("  hindi-only names: N7 skipped, keys =", len(hindi_keys))

    print("\n== address parsing ==")
    # RAW addresses taken from train/test: comma components, shuffled order,
    # no US ZIPs, Indian PINs written "600 102", French rows have no postcode
    cases = [
        # (raw address, country, expected pincode, expected state, expected city)
        ("1795 Westchester Drive, High Point, NC", "US", "", "north carolina", "high point"),
        ("105 ELM ST, MORGANTON, NC", "US", "", "north carolina", "morganton"),
        # 5-digit house number is NOT a ZIP
        ("17560 Ellis Road, Tahlequah, OK", "US", "", "oklahoma", "tahlequah"),
        ("11973 Moorland Manor Court, Fl 0, Saint Louis, MO", "US", "", "missouri", "saint louis"),
        # street name containing a state name / floor "Fl" must not win
        ("6584 Delaware Crossing Street, Bldg 8, Reynoldsburg, OH", "US", "", "ohio", "reynoldsburg"),
        ("Washington, 5649 Western Avenue, DC", "US", "", "district of columbia", "washington"),
        ("265 Chrestman Lane, TX, Combine", "US", "", "texas", "combine"),
        ("Riverhead, NY, 189 Trout Brook Lane", "US", "", "new york", "riverhead"),
        ("Mack Rd, Haltom City, Texas", "US", "", "texas", "haltom city"),
        ("2742 8th Terrace, Kansas City, KS", "US", "", "kansas", "kansas city"),
        ("12 Main St, Springfield, MA 01104", "US", "01104", "massachusetts", "springfield"),
        # city named like its state; city whose name ends in a street word
        ("New York, NY, 162 105 Street", "US", "", "new york", "new york"),
        ("12 Boardwalk, Pike Road, AL", "US", "", "alabama", "pike road"),
        ("6505 225th Place, Mountlake Terrace, WA", "US", "", "washington", "mountlake terrace"),
        ("H No 429 Mig Gh 14 Paschim Vihar, New Delhi, Delhi", "India", "", "delhi", "new delhi"),
        ("Delhi, New Delhi, 25, Extension 2B Nangloi", "India", "", "delhi", "new delhi"),
        # native-script state components map to the English state
        ("#708 FLAT NO. A 13 TIRUPATI COMPLEX GORAKHPUR, JABALPUR, मध्य प्रदेश", "India", "", "madhya pradesh", "jabalpur"),
        ("10 PRINCEP STREET, KOLKATA, পশ্চিমবঙ্গ", "India", "", "west bengal", "kolkata"),
        ("H NO - 263, SULEBHAVI VILLAGE, HUNGUND, ಕರ್ನಾಟಕ", "India", "", "karnataka", "hungund"),
        ("NO 6-3-347/6/N. , 5TH FR, N.V PLAZA, PANJAGUTTA, NAMPALLY, తెలంగాణ", "India", "", "telangana", "nampally"),
        ("12 MG Road, Pune, महाराष्ट्र", "India", "", "maharashtra", "pune"),
        # Indian state abbreviations / variants
        ("12 MG Road, Pune, MH", "India", "", "maharashtra", "pune"),
        ("LR Tower, South Janatha Road, Ernakulam, Keralam", "India", "", "kerala", "ernakulam"),
        ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India", "", "delhi", "west delhi"),
        ("L Block157 I Main Road, Anna Nagar, Chennai 600 102, Tamil Nadu", "India", "600102", "tamil nadu", "chennai"),
        ("Plot No. M-5/8, Midc, Kupwad, Miraj, Sangli, Maharashtra", "India", "", "maharashtra", "sangli"),
        ("Door No1/61, West Street, Kilangadu, Pudukkottai, Sivaganga, Tamil Nadu", "India", "", "tamil nadu", "sivaganga"),
        # France: region column, departments map to their region, city found
        # in any position, St-/St.- folded to saint
        ("43 Rue des Phalempins, Tourcoing, Hauts-de-France", "France", "", "hauts de france", "tourcoing"),
        ("Hauts-de-France, Dunkerque, 71 Rue de la Paix", "France", "", "hauts de france", "dunkerque"),
        ("Loire-Atlantique, 19 BLVD DE LA PILOTIERE, NANTES", "France", "", "pays de la loire", "nantes"),
        ("3 R. DE LA PAIX, PESSAC, Gironde", "France", "", "nouvelle aquitaine", "pessac"),
        ("Nº 19 R. Verlaine, Lille", "France", "", "", "lille"),
        ("Hauts-de-France, Avenue du Parc des Sports, Bat D, Roubaix, AV Fleming", "France", "", "hauts de france", "roubaix"),
        ("1 Avenue Mgellan, Lège-cap-ferret", "France", "", "", "lege cap ferret"),
        ("ST.-NAZAIRE, 5 Rue Jean Jaurès, Pays de la Loire", "France", "", "pays de la loire", "saint nazaire"),
        ("9 Rue George Sand, Saint-Nazaire, Pays de la Loire", "France", "", "pays de la loire", "saint nazaire"),
        ("12 Rue de la Paix, 75002 Paris", "France", "75002", "", "paris"),
        # PO box numbers (BP / CS) are not postcodes; glued "59160LOMME" is
        ("12 BIS BD FRANCOIS BLANCHO, BP 40413, NANTES, Pays de la Loire", "France", "", "pays de la loire", "nantes"),
        ("11 RUE DE PRINQUIAU, BP82228, NANTES", "France", "", "", "nantes"),
        ("13 R. G.CLEMENCEAU, 59160LOMME, LILLE, Hauts-de-France", "France", "59160", "hauts de france", "lille"),
    ]
    for addr, hint, exp_pin, exp_state, exp_city in cases:
        p = pp.parse_address(addr, hint)
        status = "OK " if (p["pincode"], p["state"], p["city"]) == \
            (exp_pin, exp_state, exp_city) else "BAD"
        print(f"  {status} {addr[:45]:<45} [{hint:<6}] -> pin={p['pincode']:<7} state={p['state']:<18} city={p['city']}")
    for addr, hint, exp_pin, exp_state, exp_city in cases:
        p = pp.parse_address(addr, hint)
        assert (p["pincode"], p["state"], p["city"]) == \
            (exp_pin, exp_state, exp_city), (addr, p)
    # empty address safe
    assert pp.parse_address("") == {"pincode": "", "state": "", "city": ""}
    print("  all address cases OK")

    print("\n== full preprocess on tiny frames ==")
    combined = pp.preprocess({"S1": src1, "S2": src2}, gt)
    assert list(combined.columns) == [
        "entity_id", "source", "name_raw", "address_raw", "name_norm",
        "address_norm", "pincode", "state", "city", "name_ngrams",
        "address_ngrams", "name_ngram_count",
        "address_ngram_count", "name_phonetic", "canopy_ids", "country",
        "ground_truth"]
    row = combined[combined.entity_id == "S1-1"].iloc[0]
    assert row["ground_truth"] == "S2-1"
    assert (combined[combined.source == "S2"]["ground_truth"] == "").all()
    print(combined[["entity_id", "source", "name_norm", "ground_truth"]].to_string())

    out = Path(tempfile.gettempdir()) / "preproc_smoke.csv"
    combined.to_csv(out, index=False, encoding="utf-8")
    print("\nCSV round-trip: wrote", out, out.stat().st_size, "bytes")
    back = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(back) == len(combined)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
