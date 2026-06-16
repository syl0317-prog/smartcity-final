import os
import json
from pathlib import Path
import pandas as pd

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
    # 3. Land Use Analysis
    # -------------------------------------------------------------
    print("[*] Processing land use CSV files (in chunks)...")
    pangyo_lu_file = workspace_dir / "토지이용" / "AL_D155_41_20241204" / "AL_D155_41_20241204.csv"
    cheongna_lu_file = workspace_dir / "토지이용" / "AL_D155_28_20241204" / "AL_D155_28_20241204.csv"
    
    pangyo_lu_codes = [4113510800, 4113510900, 4113511000, 4113511500, 4113511600, 4113511700, 4113511800]
    cheongna_lu_codes = [2826012200]
    
    def process_land_use_zones(file_path, target_codes):
        zone_counts = {}
        if not file_path.exists():
            print(f"  [!] Warning: Land use file {file_path} not found.")
            return zone_counts
            
        for chunk in pd.read_csv(file_path, usecols=['법정동코드', '용도지역지구명'], chunksize=500000, encoding='cp949'):
            filtered = chunk[chunk['법정동코드'].isin(target_codes)]
            if len(filtered) > 0:
                counts = filtered['용도지역지구명'].value_counts()
                for zone, count in counts.items():
                    zone_counts[zone] = zone_counts.get(zone, 0) + int(count)
        return zone_counts

    pangyo_lu_raw = process_land_use_zones(pangyo_lu_file, pangyo_lu_codes)
    cheongna_lu_raw = process_land_use_zones(cheongna_lu_file, cheongna_lu_codes)
    
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
    # 4. SGIS Demographics
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
    
    # -------------------------------------------------------------
    # 5. Save Combined Comparison Data JSON
    # -------------------------------------------------------------
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
    with open(comparison_output, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved comparison data to {comparison_output}")
    
    print("[+] Preprocessing successfully completed!")

if __name__ == "__main__":
    main()
