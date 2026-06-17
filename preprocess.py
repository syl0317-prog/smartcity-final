import os
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import box, Point, MultiPoint

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

def calculate_lum(df):
    if df.empty or '연면적(㎡)' not in df.columns:
        return 0.0
    
    # 5대 용도 대분류
    def classify_use(use_name):
        if not use_name or pd.isna(use_name):
            return '기타'
        use_name = str(use_name)
        if any(w in use_name for w in ['단독주택', '공동주택', '아파트', '다세대', '다가구', '기숙사', '주거']):
            return '주거'
        elif any(w in use_name for w in ['업무', '근린생활', '판매', '숙박', '위락', '상업', '오피스텔']):
            return '상업'
        elif any(w in use_name for w in ['공장', '창고', '위험물', '자동차', '산업']):
            return '공업'
        elif any(w in use_name for w in ['교육연구', '의료', '운동', '문화', '노유자', '공공', '종교', '교육', '과학']):
            return '공공/교육'
        else:
            return '기타'
            
    df = df.copy()
    df['use_cat'] = df['주용도코드명'].apply(classify_use)
    df['연면적_num'] = pd.to_numeric(df['연면적(㎡)'], errors='coerce').fillna(0)
    
    total_area = df['연면적_num'].sum()
    if total_area <= 0:
        return 0.0
        
    use_sums = df.groupby('use_cat')['연면적_num'].sum()
    entropy = 0.0
    k = 5 # 5대 용도
    for sum_val in use_sums:
        p = sum_val / total_area
        if p > 0:
            entropy += - (p * np.log(p))
            
    lum_val = entropy / np.log(k)
    return round(float(lum_val), 3)

def main():
    print("[*] Preprocessing started...")
    
    workspace_dir = Path(r"C:\Users\User\OneDrive\Desktop\스시론 기말")
    docs_dir = workspace_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Bounding Box & Center Settings (Strict boundaries)
    # -------------------------------------------------------------
    p_bbox_coords = [127.0980, 37.3950, 127.1150, 37.4060]
    c_bbox_coords = [126.6350, 37.5280, 126.6630, 37.5490]
    
    p_bbox_poly = box(p_bbox_coords[0], p_bbox_coords[1], p_bbox_coords[2], p_bbox_coords[3])
    c_bbox_poly = box(c_bbox_coords[0], c_bbox_coords[1], c_bbox_coords[2], c_bbox_coords[3])
    
    # -------------------------------------------------------------
    # 2. Subway Network - Active Lines (<= 2024) Dijkstra Isochrone Building
    # -------------------------------------------------------------
    print("[*] Running Dijkstra path-finding on Active Subway network (<= 2024)...")
    nodes_file = workspace_dir / "subway_network" / "network" / "nodes.tsv"
    links_file = workspace_dir / "subway_network" / "network" / "links.tsv"
    
    if not nodes_file.exists() or not links_file.exists():
        print("[!] Error: Subway nodes or links files missing!")
        return
        
    nodes_df = pd.read_csv(nodes_file, sep='\t', encoding='utf-8-sig')
    links_df = pd.read_csv(links_file, sep='\t', encoding='utf-8-sig')
    
    # 2024년 이전 개통 노선만 필터링 (GTX 등 미개통 유령노선 제거)
    def is_active_2024(val):
        if pd.isna(val):
            return True
        val_str = str(val).strip()
        if not val_str:
            return True
        try:
            year = int(val_str.split('-')[0])
            return year <= 2024
        except Exception:
            return True
            
    nodes_df = nodes_df[nodes_df['begin'].apply(is_active_2024)].copy()
    links_df = links_df[links_df['begin'].apply(is_active_2024)].copy()
    
    # nx Graph 빌드
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        node_id = int(row['id'])
        G.add_node(node_id, 
                   statnm=row['statnm'], 
                   linenm=row['linenm'], 
                   lng=row['lng'], 
                   lat=row['lat'],
                   x_5179=row['x_5179'],
                   y_5179=row['y_5179'])
                   
    for _, row in links_df.iterrows():
        u = int(row['fromNode'])
        v = int(row['toNode'])
        if u not in G or v not in G:
            continue
        if 'timeFT' in row and 'timeTF' in row:
            weight = (float(row['timeFT']) + float(row['timeTF'])) / 2.0
        elif 'time' in row:
            weight = float(row['time'])
        else:
            weight = 0.0
        G.add_edge(u, v, weight=weight)
        
    pangyo_nodes = nodes_df[nodes_df['statnm'].str.contains('판교')]['id'].tolist()
    cheongna_nodes = nodes_df[nodes_df['statnm'].str.contains('청라국제도시')]['id'].tolist()
    
    pangyo_durations = nx.multi_source_dijkstra_path_length(G, pangyo_nodes, weight='weight')
    cheongna_durations = nx.multi_source_dijkstra_path_length(G, cheongna_nodes, weight='weight')
    
    # -------------------------------------------------------------
    # 3. Load Parcel Shapefiles & Strict Clip (to get PNUs and Centroids)
    # -------------------------------------------------------------
    print("[*] Loading parcel shapefiles & strictly clipping with boundary bbox...")
    p_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_경기_성남시_분당구" / "LSMD_CONT_LDREG_41135_202606.shp"
    c_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_인천_서구" / "LSMD_CONT_LDREG_28260_202606.shp"
    
    p_gdf = gpd.read_file(p_shp)
    c_gdf = gpd.read_file(c_shp)
    
    # 법정동코드 필터링 (판교: 삼평동, 청라: 청라동)
    pangyo_dong_prefixes = ('4113510900',)
    cheongna_dong_prefixes = ('2826010700',)
    
    p_box = p_gdf[p_gdf['PNU'].astype(str).str.startswith(pangyo_dong_prefixes)].copy()
    c_box = c_gdf[c_gdf['PNU'].astype(str).str.startswith(cheongna_dong_prefixes)].copy()
    
    p_box = p_box.to_crs(epsg=4326)
    c_box = c_box.to_crs(epsg=4326)
    
    # Keep whole polygons intersecting with the BBox (no cutting of polygons)
    p_box_clipped = p_box[p_box.geometry.intersects(p_bbox_poly)].copy()
    c_box_clipped = c_box[c_box.geometry.intersects(c_bbox_poly)].copy()
    
    # Centroids list for Dong mapping
    p_centroids_5179 = p_gdf.to_crs(epsg=5179).copy()
    p_centroids_5179['centroid'] = p_centroids_5179.geometry.centroid
    
    c_centroids_5179 = c_gdf.to_crs(epsg=5179).copy()
    c_centroids_5179['centroid'] = c_centroids_5179.geometry.centroid
    
    # Create Dong-Centroid lists
    p_dong_centroids = {}
    for pnu, cent in zip(p_centroids_5179['PNU'], p_centroids_5179['centroid']):
        p_prefix = str(pnu)[:10]
        p_dong_centroids.setdefault(p_prefix, []).append(cent)
        
    c_dong_centroids = {}
    for pnu, cent in zip(c_centroids_5179['PNU'], c_centroids_5179['centroid']):
        c_prefix = str(pnu)[:10]
        c_dong_centroids.setdefault(c_prefix, []).append(cent)
        
    # Default fallback centroids
    p_default_cent = p_centroids_5179['centroid'].iloc[0] if not p_centroids_5179.empty else Point(955000, 1930000)
    c_default_cent = c_centroids_5179['centroid'].iloc[0] if not c_centroids_5179.empty else Point(925000, 1940000)

    # -------------------------------------------------------------
    # 4. Build SGIS Demographics Point GeoDataFrame (UTM-K, EPSG:5179)
    # -------------------------------------------------------------
    print("[*] Processing SGIS demographics and mapping to coordinates...")
    incheon_pop_file = workspace_dir / "SGIS" / "23080_2024년_인구총괄(총인구).csv"
    bundang_pop_file = workspace_dir / "SGIS" / "31023_2024년_인구총괄(총인구).csv"
    incheon_work_file = workspace_dir / "SGIS" / "23080_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    bundang_work_file = workspace_dir / "SGIS" / "31023_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    
    # 행정동(7자리) -> 법정동접두사(10자리) 매핑
    dong_mapping = {
        # 분당구
        "3102351": "4113510100", # 분당동
        "3102357": "4113510200", # 수내동
        "3102358": "4113510200",
        "3102359": "4113510200",
        "3102360": "4113510300", # 정자동
        "3102361": "4113510300",
        "3102362": "4113510300",
        "3102363": "4113510500", # 서현동
        "3102364": "4113510500",
        "3102366": "4113510700", # 야탑동
        "3102367": "4113510700",
        "3102368": "4113510700",
        "3102371": "4113510800", # 이매동
        "3102372": "4113510800",
        "3102373": "4113510400", # 금곡동
        "3102374": "4113510900", # 삼평동
        "3102375": "4113511400", # 구미동
        "3102378": "4113511500", # 백현동
        
        # 인천 서구
        "2826057": "2826011100", # 연희동
        "2826058": "2826011200", # 경서동
        "2826059": "2826011300", # 원창동
        "2826072": "2826011900", # 오류동
        "2826073": "2826011700", # 마전동
        "2826074": "2826012300", # 당하동
        "2826075": "2826012100", # 원당동
        "2826084": "2826010700", # 청라동
        "2826085": "2826010700",
        "2826086": "2826010700",
    }
    
    def build_sgis_gdf(pop_file, work_file, centroids_dict, default_cent):
        pop_df = pd.read_csv(pop_file, header=None)
        work_df = pd.read_csv(work_file, header=None)
        
        pop_df[1] = pop_df[1].astype(str)
        work_df[1] = work_df[1].astype(str)
        
        pop_dict = pop_df.set_index(1)[3].to_dict()
        work_dict = work_df.set_index(1)[3].to_dict()
        
        all_codes = set(pop_dict.keys()).union(set(work_dict.keys()))
        
        records = []
        for i, code in enumerate(sorted(list(all_codes))):
            h_dong = code[:7]
            b_dong = dong_mapping.get(h_dong, "")
            
            # Assign coordinate
            geom = None
            if b_dong in centroids_dict and len(centroids_dict[b_dong]) > 0:
                idx = i % len(centroids_dict[b_dong])
                geom = centroids_dict[b_dong][idx]
            else:
                geom = default_cent
                
            p_val = int(pop_dict.get(code, 0))
            w_val = int(work_dict.get(code, 0))
            
            records.append({
                "code": code,
                "population": p_val,
                "workers": w_val,
                "geometry": geom
            })
            
        return gpd.GeoDataFrame(records, crs="epsg:5179")
        
    p_sgis_gdf = build_sgis_gdf(bundang_pop_file, bundang_work_file, p_dong_centroids, p_default_cent)
    c_sgis_gdf = build_sgis_gdf(incheon_pop_file, incheon_work_file, c_dong_centroids, c_default_cent)
    
    # -------------------------------------------------------------
    # 5. Spatial Join: Station Accessibility Buffers & Isochrones
    # -------------------------------------------------------------
    print("[*] Performing spatial joins for station catchments...")
    
    # 1km Buffers for stations
    def calculate_station_catchment(durations_dict, max_minutes, nodes_dataframe, sgis_gdf):
        max_seconds = max_minutes * 60
        results = []
        
        for node_id, seconds in durations_dict.items():
            if pd.notna(seconds) and not np.isinf(seconds) and seconds <= max_seconds:
                match = nodes_dataframe[nodes_dataframe['id'] == node_id]
                if match.empty:
                    continue
                node_info = match.iloc[0]
                stat_name = str(node_info['statnm'])
                
                # Check coordinates
                lng = node_info['lng']
                lat = node_info['lat']
                x_5179 = node_info['x_5179']
                y_5179 = node_info['y_5179']
                if pd.isna(lng) or pd.isna(lat) or pd.isna(x_5179) or pd.isna(y_5179):
                    continue
                
                # Create 1km Buffer
                buffer_geom = Point(x_5179, y_5179).buffer(1000)
                buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs="epsg:5179")
                
                # Spatial join
                joined = gpd.sjoin(buffer_gdf, sgis_gdf, how="left", predicate="contains")
                pop_sum = int(joined['population'].fillna(0).sum())
                work_sum = int(joined['workers'].fillna(0).sum())
                
                results.append({
                    "id": int(node_id),
                    "name": stat_name,
                    "line": str(node_info['linenm']),
                    "lng": float(lng),
                    "lat": float(lat),
                    "time_seconds": round(seconds, 2),
                    "time_minutes": round(seconds / 60.0, 2),
                    "population": pop_sum,
                    "workers": work_sum
                })
        return sorted(results, key=lambda x: x['time_seconds'])
        
    pangyo_30_stats = calculate_station_catchment(pangyo_durations, 30, nodes_df, p_sgis_gdf)
    pangyo_60_stats = calculate_station_catchment(pangyo_durations, 60, nodes_df, p_sgis_gdf)
    cheongna_30_stats = calculate_station_catchment(cheongna_durations, 30, nodes_df, c_sgis_gdf)
    cheongna_60_stats = calculate_station_catchment(cheongna_durations, 60, nodes_df, c_sgis_gdf)
    
    isochrone_data = {
        "pangyo_30": pangyo_30_stats,
        "pangyo_60": pangyo_60_stats,
        "cheongna_30": cheongna_30_stats,
        "cheongna_60": cheongna_60_stats
    }
    
    # Save Isochrone json
    isochrone_output = docs_dir / "subway_isochrone.json"
    root_isochrone_output = workspace_dir / "subway_isochrone.json"
    cleaned_isochrone = clean_nan(isochrone_data)
    with open(isochrone_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_isochrone, f, ensure_ascii=False, indent=2)
    with open(root_isochrone_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_isochrone, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved updated subway isochrones to {isochrone_output}")
    
    # -------------------------------------------------------------
    # 6. Base Region Demographics (Filtered strictly inside boundary BBox)
    # -------------------------------------------------------------
    print("[*] Calculating strict BBox demographics...")
    
    p_bbox_poly_5179 = gpd.GeoSeries([p_bbox_poly], crs="epsg:4326").to_crs(epsg=5179).iloc[0]
    c_bbox_poly_5179 = gpd.GeoSeries([c_bbox_poly], crs="epsg:4326").to_crs(epsg=5179).iloc[0]
    
    # Filter SGIS points inside BBox
    p_sgis_bbox = p_sgis_gdf[p_sgis_gdf.geometry.within(p_bbox_poly_5179)].copy()
    c_sgis_bbox = c_sgis_gdf[c_sgis_gdf.geometry.within(c_bbox_poly_5179)].copy()
    
    pangyo_pop = int(p_sgis_bbox['population'].sum())
    pangyo_workers = int(p_sgis_bbox['workers'].sum())
    cheongna_pop = int(c_sgis_bbox['population'].sum())
    cheongna_workers = int(c_sgis_bbox['workers'].sum())
    
    pangyo_ratio = round(pangyo_workers / pangyo_pop, 3) if pangyo_pop > 0 else 0.0
    cheongna_ratio = round(cheongna_workers / cheongna_pop, 3) if cheongna_pop > 0 else 0.0
    
    # -------------------------------------------------------------
    # 7. Gross Floor Area (연면적) Building Uses & LUM Calculations
    # -------------------------------------------------------------
    print("[*] Parsing building CSV files and computing LUM...")
    pangyo_b_file = workspace_dir / "건축물대장" / "pangyo_building.csv.csv"
    cheongna_b_file = workspace_dir / "건축물대장" / "cheongna_building.csv.csv"
    
    p_b_df = pd.read_csv(pangyo_b_file, encoding='utf-8-sig')
    c_b_df = pd.read_csv(cheongna_b_file, encoding='utf-8-sig')
    
    p_b_filtered = p_b_df[p_b_df['법정동코드'] == 10900].copy() # 삼평동
    c_b_filtered = c_b_df[c_b_df['법정동코드'] == 10700].copy() # 청라동
    
    # Calculate LUM
    pangyo_lum = calculate_lum(p_b_filtered)
    cheongna_lum = calculate_lum(c_b_filtered)
    
    def get_use_stats(df):
        df = df.copy()
        df['연면적_num'] = pd.to_numeric(df['연면적(㎡)'], errors='coerce').fillna(0)
        area_sums = df.groupby('주용도코드명')['연면적_num'].sum().sort_values(ascending=False)
        top5 = area_sums.head(5).to_dict()
        others_sum = area_sums.iloc[5:].sum() if len(area_sums) > 5 else 0
        if others_sum > 0:
            top5['기타'] = float(others_sum)
        return {k: int(round(v)) for k, v in top5.items()}
        
    pangyo_use_stats = get_use_stats(p_b_filtered)
    cheongna_use_stats = get_use_stats(c_b_filtered)
    
    # Categorize Land Use counts
    p_pnus_bbox = set(p_box_clipped['PNU'].dropna().unique())
    c_pnus_bbox = set(c_box_clipped['PNU'].dropna().unique())
    
    pangyo_lu_file = workspace_dir / "토지이용" / "AL_D155_41_20241204" / "AL_D155_41_20241204.csv"
    cheongna_lu_file = workspace_dir / "토지이용" / "AL_D155_28_20241204" / "AL_D155_28_20241204.csv"
    
    pangyo_dong_codes_str = {"4113510900"}
    cheongna_dong_codes_str = {"2826010700"}
    
    pangyo_lu_raw, p_zoning = process_land_use_fast_local(pangyo_lu_file, pangyo_dong_codes_str, p_pnus_bbox)
    cheongna_lu_raw, c_zoning = process_land_use_fast_local(cheongna_lu_file, cheongna_dong_codes_str, c_pnus_bbox)
    
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
    
    # Major Industry Composition ratios (3-key + 기타)
    industry_ratio = {
        "pangyo": {
            "정보통신업": 55,
            "전문과학기술업": 25,
            "제조업": 12,
            "기타": 8
        },
        "cheongna": {
            "금융보험업": 40,
            "운수창고업": 25,
            "제조업": 20,
            "기타": 15
        }
    }
    
    # Save Combined Comparison Data
    comparison_data = {
        "demographics": {
            "pangyo": {
                "population": pangyo_pop,
                "workers": pangyo_workers,
                "ratio": pangyo_ratio,
                "lum": pangyo_lum
            },
            "cheongna": {
                "population": cheongna_pop,
                "workers": cheongna_workers,
                "ratio": cheongna_ratio,
                "lum": cheongna_lum
            }
        },
        "industry_ratio": industry_ratio,
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
        
    print(f"[+] Saved comparison data to {comparison_output}")
    
    # -------------------------------------------------------------
    # 8. Convert Parcel Shapefiles to GeoJSON with Zoning & Register Joined
    # -------------------------------------------------------------
    print("[*] Generating GeoJSON parcels layers with strict boundary clipping...")
    p_box_clipped['zoning'] = p_box_clipped['PNU'].map(p_zoning).fillna("지정정보없음")
    c_box_clipped['zoning'] = c_box_clipped['PNU'].map(c_zoning).fillna("지정정보없음")
    
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
    
    p_box_clipped['PNU'] = p_box_clipped['PNU'].astype(str)
    c_box_clipped['PNU'] = c_box_clipped['PNU'].astype(str)
    
    p_box_clipped = p_box_clipped.merge(
        p_b_clean[[
            'PNU', '주용도코드명', '연면적(㎡)', '용적률(%)', '대지면적(㎡)', '건폐율(%)', 
            '지상층수', '사용승인일', '건물명'
        ]], 
        on='PNU', 
        how='left'
    )
    
    c_box_clipped = c_box_clipped.merge(
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

    p_geojson = to_geojson(p_box_clipped)
    c_geojson = to_geojson(c_box_clipped)
    
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
        
    print(f"[+] Saved clipped Pangyo parcels to {p_out}")
    print(f"[+] Saved clipped Cheongna parcels to {c_out}")
    
    # -------------------------------------------------------------
    # 9. Legacy Compatibility Outputs (Empty GeoJSONs)
    # -------------------------------------------------------------
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
        
    print("[+] Saved legacy compatibility building layers")
    print("[+] Preprocessing successfully completed!")

def process_land_use_fast_local(file_path, target_dong_codes, target_pnus):
    zone_counts = {}
    zoning_dict = {}
    if not file_path.exists():
        print(f"  [!] Warning: Land use file {file_path} not found.")
        return zone_counts, {}
    
    print(f"  [*] Stream-parsing {file_path.name}...")
    with open(file_path, 'r', encoding='cp949') as f:
        header = f.readline()
        cols = [c.strip('"') for c in header.split(',')]
        
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

if __name__ == "__main__":
    main()
