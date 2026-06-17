import re, os

files = [
    "docs/cheongna_buildings.geojson",
    "docs/cheongna_parcels.geojson", 
    "docs/pangyo_buildings.geojson",
    "docs/pangyo_parcels.geojson",
    "docs/subway_isochrone.json",
    "docs/comparison_data.json",
    "docs/cumulative_accessibility.json"
]

for f in files:
    if not os.path.exists(f):
        print(f"{f} 파일이 존재하지 않습니다. 건너뜁니다.")
        continue
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    fixed = re.sub(r'\bNaN\b', 'null', content)
    fixed = re.sub(r'\bInfinity\b', 'null', fixed)
    fixed = re.sub(r'\b-Infinity\b', 'null', fixed)
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write(fixed)
    print(f"{f} 완료")
