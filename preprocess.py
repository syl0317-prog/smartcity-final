import os
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

def clean_nan(obj):
    import math
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif pd.isna(obj):
        return None
    return obj

def process_land_use_fast(file_path, target_dong_codes, target_pnus):
    """
    Reads a large land use CSV file line by line in CP949 encoding.
    Returns:
      zone_counts: dict of {zone_name: count} for target_dong_codes
      zoning_dict: dict of {pnu: zoning_str} for target_pnus
    """
    zone_counts = {}
    zoning_dict = {}
    if not file_path.exists():
        print(f"  [!] Warning: Land use file {file_path} not found.")
        return zone_counts, zoning_dict
    
    print(f"  [*] Stream-parsing {file_path.name}...")
    with open(file_path, 'r', encoding='cp949') as f:
        header = f.readline()
        cols = [c.strip('"') for c in header.split(',')]
        
        # We need: '고유번호' (pnu), '법정동코드' (dong), '용도지역지구명' (zone), '용도지역지구코드' (code)
        idx_pnu = cols.index('고유번호') if '고유번호' in cols else 0
        idx_dong = cols.index('법정동코드') if '법정동코드' in cols else 1
        idx_zone = cols.index('용도지역지구명') if '용도지역지구명' in cols else 10
        idx_code = cols.index('용도지역지구코드') if '용도지역지구코드' in cols else 9
        
        for line in f:
            row = line.split(',')
            if len(row) > max(idx_pnu, idx_dong, idx_zone, idx_code):
                dong = row[idx_dong].strip('"')
                if dong in target_dong_codes:
                    zone = row[idx_zone].strip('"')
                    pnu = row[idx_pnu].strip('"')
                    code = row[idx_code].strip('"')
                    if zone:
                        zone_counts[zone] = zone_counts.get(zone, 0) + 1
                        if code.startswith('UQA'):
                            if pnu in target_pnus:
                                zoning_dict.setdefault(pnu, set()).add((code, zone))
                            
    def get_priority_key(item):
        c, n = item
        is_detailed = 1 if c not in ['UQA001', 'UQA002', 'UQA003', 'UQA004', 'UQA000'] else 0
        name_len = len(n) if n else 0
        return (is_detailed, name_len, c)

    zoning_str_dict = {}
    for k, v in zoning_dict.items():
        sorted_items = sorted(list(v), key=get_priority_key, reverse=True)
        zoning_str_dict[k] = ", ".join([item[1] for item in sorted_items])
        
    return zone_counts, zoning_str_dict

def main():
    print("[*] Preprocessing started...")
    
    # Define directories
    workspace_dir = Path(r"C:\Users\User\OneDrive\Desktop\스시론 기말")
    docs_dir = workspace_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Subway Network - Copy Original subway_isochrone.json
    # -------------------------------------------------------------
    print("[*] Copying original subway_isochrone.json to docs/...")
    original_isochrone_path = workspace_dir / "subway_isochrone.json"
    if not original_isochrone_path.exists():
        original_isochrone_path = workspace_dir / "subway_network" / "network" / "subway_isochrone.json"
        
    if not original_isochrone_path.exists():
        print("[!] Error: subway_isochrone.json not found!")
        return
        
    with open(original_isochrone_path, 'r', encoding='utf-8') as f:
        isochrone_data = json.load(f)
        
    isochrone_output = docs_dir / "subway_isochrone.json"
    with open(isochrone_output, 'w', encoding='utf-8') as f:
        json.dump(isochrone_data, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved original subway isochrones to {isochrone_output}")
    
    # -------------------------------------------------------------
    # 2. Building Data Processing (Statistics Only)
    # -------------------------------------------------------------
    print("[*] Processing building CSV files...")
    pangyo_b_file = workspace_dir / "건축물대장" / "pangyo_building.csv.csv"
    cheongna_b_file = workspace_dir / "건축물대장" / "cheongna_building.csv.csv"
    
    p_b_df = pd.read_csv(pangyo_b_file, encoding='utf-8-sig')
    c_b_df = pd.read_csv(cheongna_b_file, encoding='utf-8-sig')
    
    pangyo_dong_codes = [10800, 10900, 11000, 11500, 11600, 11700, 11800]
    cheongna_dong_codes = [12200]
    
    p_b_filtered = p_b_df[p_b_df['법정동코드'].isin(pangyo_dong_codes)].copy()
    c_b_filtered = c_b_df[c_b_df['법정동코드'].isin(cheongna_dong_codes)].copy()
    
    def get_use_stats(df):
        counts = df['주용도코드명'].value_counts()
        top5 = counts.head(5).to_dict()
        others_sum = counts.iloc[5:].sum() if len(counts) > 5 else 0
        if others_sum > 0:
            top5['기타'] = int(others_sum)
        return {k: int(v) for k, v in top5.items()}
    
    pangyo_use_stats = get_use_stats(p_b_filtered)
    cheongna_use_stats = get_use_stats(c_b_filtered)
    
    # -------------------------------------------------------------
    # 3. Load Parcel Shapefiles & BBox Filter (to get PNUs first)
    # -------------------------------------------------------------
    print("[*] Loading parcel shapefiles...")
    p_box_coords = [127.0980, 37.3950, 127.1150, 37.4060]
    c_box_coords = [126.6350, 37.5280, 126.6630, 37.5490]
    
    # Pangyo Parcels
    p_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_경기_성남시_분당구" / "LSMD_CONT_LDREG_41135_202606.shp"
    p_gdf = gpd.read_file(p_shp)
    pangyo_dong_prefixes = ('4113510800', '4113510900', '4113511000', '4113511500', '4113511600', '4113511700', '4113511800')
    p_box = p_gdf[p_gdf['PNU'].astype(str).str.startswith(pangyo_dong_prefixes)].copy()
    p_box = p_box.to_crs(epsg=4326)
    p_box = p_box.cx[p_box_coords[0]:p_box_coords[2], p_box_coords[1]:p_box_coords[3]].copy()
    # Filter by centroid to strictly keep parcels inside the bbox
    p_centroids = p_box.geometry.centroid
    p_box = p_box[
        (p_centroids.x >= p_box_coords[0]) & 
        (p_centroids.x <= p_box_coords[2]) & 
        (p_centroids.y >= p_box_coords[1]) & 
        (p_centroids.y <= p_box_coords[3])
    ].copy()
    
    # Cheongna Parcels
    c_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_인천_서구" / "LSMD_CONT_LDREG_28260_202606.shp"
    c_gdf = gpd.read_file(c_shp)
    cheongna_dong_prefixes = ('2826012200',)
    c_box = c_gdf[c_gdf['PNU'].astype(str).str.startswith(cheongna_dong_prefixes)].copy()
    c_box = c_box.to_crs(epsg=4326)
    c_box = c_box.cx[c_box_coords[0]:c_box_coords[2], c_box_coords[1]:c_box_coords[3]].copy()
    # Filter by centroid to strictly keep parcels inside the bbox
    c_centroids = c_box.geometry.centroid
    c_box = c_box[
        (c_centroids.x >= c_box_coords[0]) & 
        (c_centroids.x <= c_box_coords[2]) & 
        (c_centroids.y >= c_box_coords[1]) & 
        (c_centroids.y <= c_box_coords[3])
    ].copy()
    
    p_pnus = set(p_box['PNU'].dropna().unique())
    c_pnus = set(c_box['PNU'].dropna().unique())
    
    # -------------------------------------------------------------
    # 4. Land Use Analysis + Zoning Matching (Fast Stream Parse)
    # -------------------------------------------------------------
    print("[*] Processing land use and zoning (fast stream)...")
    pangyo_lu_file = workspace_dir / "토지이용" / "AL_D155_41_20241204" / "AL_D155_41_20241204.csv"
    cheongna_lu_file = workspace_dir / "토지이용" / "AL_D155_28_20241204" / "AL_D155_28_20241204.csv"
    
    pangyo_dong_codes_str = {"4113510800", "4113510900", "4113511000", "4113511500", "4113511600", "4113511700", "4113511800"}
    cheongna_dong_codes_str = {"2826012200"}
    
    pangyo_lu_raw, p_zoning = process_land_use_fast(pangyo_lu_file, pangyo_dong_codes_str, p_pnus)
    cheongna_lu_raw, c_zoning = process_land_use_fast(cheongna_lu_file, cheongna_dong_codes_str, c_pnus)
    
    def categorize_zones(zone_counts):
        categories = {
            "주거지역": 0,
            "상업지역": 0,
            "공업지역": 0,
            "녹지지역": 0,
            "개발제한구역": 0,
            "기타/지구단위": 0
        }
        for zone, count in zone_counts.items():
            if any(term in zone for term in ["주거", "주택", "준주거"]):
                categories["주거지역"] += count
            elif any(term in zone for term in ["상업", "업무"]):
                categories["상업지역"] += count
            elif any(term in zone for term in ["공업", "공장", "산업"]):
                categories["공업지역"] += count
            elif any(term in zone for term in ["녹지", "공원", "보전녹지", "자연녹지"]):
                categories["녹지지역"] += count
            elif "개발제한구역" in zone:
                categories["개발제한구역"] += count
            else:
                categories["기타/지구단위"] += count
        return categories

    pangyo_lu_categories = categorize_zones(pangyo_lu_raw)
    cheongna_lu_categories = categorize_zones(cheongna_lu_raw)
    
    # -------------------------------------------------------------
    # 5. SGIS Demographics
    # -------------------------------------------------------------
    print("[*] Processing SGIS demographics...")
    incheon_pop_file = workspace_dir / "SGIS" / "23080_2024년_인구총괄(총인구).csv"
    bundang_pop_file = workspace_dir / "SGIS" / "31023_2024년_인구총괄(총인구).csv"
    incheon_work_file = workspace_dir / "SGIS" / "23080_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    bundang_work_file = workspace_dir / "SGIS" / "31023_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    
    pangyo_sgis_codes = ["3102371", "3102372", "3102374", "3102378"]
    cheongna_sgis_codes = ["2308084", "2308085", "2308086"]
    
    def sum_sgis_value(file_path, target_prefixes):
        if not file_path.exists():
            return 0
        df = pd.read_csv(file_path, header=None)
        df[4] = df[1].astype(str).str[:7]
        filtered = df[df[4].isin(target_prefixes)]
        return int(filtered[3].sum())

    pangyo_pop = sum_sgis_value(bundang_pop_file, pangyo_sgis_codes)
    cheongna_pop = sum_sgis_value(incheon_pop_file, cheongna_sgis_codes)
    
    pangyo_workers = sum_sgis_value(bundang_work_file, pangyo_sgis_codes)
    cheongna_workers = sum_sgis_value(incheon_work_file, cheongna_sgis_codes)
    
    pangyo_ratio = round(pangyo_workers / pangyo_pop, 3) if pangyo_pop > 0 else 0
    cheongna_ratio = round(cheongna_workers / cheongna_pop, 3) if cheongna_pop > 0 else 0
    
    # Save Combined Comparison Data JSON
    comparison_data = {
        "demographics": {
            "pangyo": {
                "population": pangyo_pop,
                "workers": pangyo_workers,
                "ratio": pangyo_ratio
            },
            "cheongna": {
                "population": cheongna_pop,
                "workers": cheongna_workers,
                "ratio": cheongna_ratio
            }
        },
        "building_uses": {
            "pangyo": pangyo_use_stats,
            "cheongna": cheongna_use_stats
        },
        "land_use_zones": {
            "pangyo": pangyo_lu_categories,
            "cheongna": cheongna_lu_categories
        }
    }
    
    comparison_output = docs_dir / "comparison_data.json"
    root_comparison_output = workspace_dir / "comparison_data.json"
    cleaned_comparison = clean_nan(comparison_data)
    
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_comparison, f, ensure_ascii=False, indent=2)
    with open(root_comparison_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_comparison, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved comparison data to {comparison_output} and root")
    
    # -------------------------------------------------------------
    # 6. Convert Parcel Shapefiles to GeoJSON with Zoning & Register Joined
    # -------------------------------------------------------------
    print("[*] Merging building register details to parcels...")
    p_box['zoning'] = p_box['PNU'].map(p_zoning).fillna("지정정보없음")
    c_box['zoning'] = c_box['PNU'].map(c_zoning).fillna("지정정보없음")
    
    def build_pnu_from_register(df):
        df = df.copy()
        df['대지구분코드'] = df['대지구분코드'].fillna(0).astype(int)
        df['번'] = df['번'].fillna(0).astype(int)
        df['지'] = df['지'].fillna(0).astype(int)
        df['시군구코드'] = df['시군구코드'].astype(str)
        df['법정동코드_str'] = df['법정동코드'].astype(str).str.zfill(5)
        
        land_type = df['대지구분코드'].apply(lambda x: '2' if x == 2 else '1')
        
        pnu_series = (
            df['시군구코드'] + 
            df['법정동코드_str'] + 
            land_type + 
            df['번'].astype(str).str.zfill(4) + 
            df['지'].astype(str).str.zfill(4)
        )
        df['PNU'] = pnu_series
        return df
        
    p_b_clean = build_pnu_from_register(p_b_filtered)
    p_b_clean = p_b_clean.sort_values(by='연면적(㎡)', ascending=False).drop_duplicates(subset=['PNU'])
    
    c_b_clean = build_pnu_from_register(c_b_filtered)
    c_b_clean = c_b_clean.sort_values(by='연면적(㎡)', ascending=False).drop_duplicates(subset=['PNU'])
    
    p_box['PNU'] = p_box['PNU'].astype(str)
    c_box['PNU'] = c_box['PNU'].astype(str)
    
    p_box = p_box.merge(
        p_b_clean[[
            'PNU', '주용도코드명', '연면적(㎡)', '용적률(%)', '대지면적(㎡)', '건폐율(%)', 
            '지상층수', '사용승인일', '건물명'
        ]], 
        on='PNU', 
        how='left'
    )
    
    c_box = c_box.merge(
        c_b_clean[[
            'PNU', '주용도코드명', '연면적(㎡)', '용적률(%)', '대지면적(㎡)', '건폐율(%)', 
            '지상층수', '사용승인일', '건물명'
        ]], 
        on='PNU', 
        how='left'
    )
    
    def to_geojson(gdf):
        features = []
        for idx, row in gdf.iterrows():
            geom = row['geometry']
            pnu = row['PNU']
            jibun = row['JIBUN']
            zoning = row['zoning']
            
            # Map attributes
            use = row['주용도코드명'] if '주용도코드명' in row and pd.notna(row['주용도코드명']) else "정보없음"
            area = float(row['연면적(㎡)']) if '연면적(㎡)' in row and pd.notna(row['연면적(㎡)']) else None
            far = float(row['용적률(%)']) if '용적률(%)' in row and pd.notna(row['용적률(%)']) else None
            lot_area = float(row['대지면적(㎡)']) if '대지면적(㎡)' in row and pd.notna(row['대지면적(㎡)']) else None
            bc_ratio = float(row['건폐율(%)']) if '건폐율(%)' in row and pd.notna(row['건폐율(%)']) else None
            floors_up = int(row['지상층수']) if '지상층수' in row and pd.notna(row['지상층수']) else 0
            
            app_y = row['사용승인일'] if '사용승인일' in row else None
            if pd.notna(app_y):
                try:
                    app_y_str = str(int(float(app_y)))
                    if len(app_y_str) >= 4:
                        app_y_str = app_y_str[:4]
                    else:
                        app_y_str = "정보없음"
                except Exception:
                    app_y_str = "정보없음"
            else:
                app_y_str = "정보없음"
                
            name = row['건물명'] if '건물명' in row and pd.notna(row['건물명']) else "대지"
            
            feature = {
                "type": "Feature",
                "geometry": json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'],
                "properties": {
                    "pnu": pnu,
                    "jibun": jibun,
                    "zoning": zoning,
                    "name": name,
                    "use": use,
                    "area": area,
                    "far": far,
                    "lot_area": lot_area,
                    "bc_ratio": bc_ratio,
                    "floors_up": floors_up,
                    "approved_year": app_y_str
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    p_geojson = to_geojson(p_box)
    c_geojson = to_geojson(c_box)
    
    p_out = workspace_dir / "pangyo_parcels.geojson"
    c_out = workspace_dir / "cheongna_parcels.geojson"
    
    cleaned_p_geojson = clean_nan(p_geojson)
    cleaned_c_geojson = clean_nan(c_geojson)

    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_p_geojson, f, ensure_ascii=False, indent=2)
    with open(c_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_c_geojson, f, ensure_ascii=False, indent=2)
        
    with open(docs_dir / "pangyo_parcels.geojson", "w", encoding="utf-8") as f:
        json.dump(cleaned_p_geojson, f, ensure_ascii=False, indent=2)
    with open(docs_dir / "cheongna_parcels.geojson", "w", encoding="utf-8") as f:
        json.dump(cleaned_c_geojson, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved Pangyo parcels to {p_out} and docs/")
    print(f"[+] Saved Cheongna parcels to {c_out} and docs/")
    
    # -------------------------------------------------------------
    # 7. Building Data Processing (Deprecated - Outputting Empty GeoJSONs)
    # -------------------------------------------------------------
    print("[*] Outputting empty building geojson layers to maintain legacy compatibility...")
    empty_geojson = {"type": "FeatureCollection", "features": []}
    
    p_b_out = workspace_dir / "pangyo_buildings.geojson"
    c_b_out = workspace_dir / "cheongna_buildings.geojson"
    
    with open(p_b_out, "w", encoding="utf-8") as f:
        json.dump(empty_geojson, f, ensure_ascii=False)
    with open(c_b_out, "w", encoding="utf-8") as f:
        json.dump(empty_geojson, f, ensure_ascii=False)
        
    with open(docs_dir / "pangyo_buildings.geojson", "w", encoding="utf-8") as f:
        json.dump(empty_geojson, f, ensure_ascii=False)
    with open(docs_dir / "cheongna_buildings.geojson", "w", encoding="utf-8") as f:
        json.dump(empty_geojson, f, ensure_ascii=False)
        
    print("[+] Saved empty building geojsons to docs/ and root")
    print("[+] Preprocessing successfully completed!")

if __name__ == "__main__":
    main()
