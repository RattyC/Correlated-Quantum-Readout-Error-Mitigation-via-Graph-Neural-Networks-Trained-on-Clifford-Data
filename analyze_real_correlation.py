import json

with open("real_hw_calibration.json") as f:
    data = json.load(f)

# ใช้ prep="00" ของแต่ละคู่ (all-zero state) เหมือน calibrate_correlations.py เดิม
pairs = {}
for entry in data:
    if entry["prep_state"] == "00":
        pairs[(entry["q_a"], entry["q_b"])] = entry["counts"]

def pearson_from_counts(counts):
    total = sum(counts.values())
    # key format: "q_b q_a" (leftmost=q_b, rightmost=q_a)
    ex = ey = exy = 0.0
    for bitstr, c in counts.items():
        q_b_bit = int(bitstr[0])
        q_a_bit = int(bitstr[1])
        p = c / total
        ex += q_a_bit * p
        ey += q_b_bit * p
        exy += q_a_bit * q_b_bit * p
    varx = ex * (1 - ex)
    vary = ey * (1 - ey)
    if varx == 0 or vary == 0:
        return 0.0, ex, ey
    r = (exy - ex * ey) / (varx * vary) ** 0.5
    return r, ex, ey

print(f"{'pair':<10}{'P(q_a err)':<14}{'P(q_b err)':<14}{'Pearson r':<12}")
for (qa, qb), counts in pairs.items():
    r, ex, ey = pearson_from_counts(counts)
    print(f"({qa},{qb})   {ex:<14.4%}{ey:<14.4%}{r:<12.4f}")
