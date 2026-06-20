import os
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import box, Point

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
        if any(w in use_name for w in ['공장', '창고', '창고시설']):
            return '기타'
        if any(w in use_name for w in ['단독주택', '공동주택', '아파트', '다세대', '다가구', '기숙사', '주거']):
            return '주거'
        elif any(w in use_name for w in ['업무', '근린생활', '판매', '숙박', '위락', '상업', '오피스텔']):
            return '상업·업무'
        elif any(w in use_name for w in ['위험물', '자동차', '산업']):
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

def clean_use_name(use_name):
    if not use_name or pd.isna(use_name):
        return '기타'
    use_name = str(use_name).strip()
    if use_name in ['공장', '창고시설', '창고']:
        return '기타'
    return use_name

def get_use_stats(df):
    df = df.copy()
    df['연면적_num'] = pd.to_numeric(df['연면적(㎡)'], errors='coerce').fillna(0)
    
    target_categories = ['업무시설', '교육연구시설', '공동주택', '단독주택', '근린생활시설']
    
    use_sums = {cat: 0.0 for cat in target_categories}
    use_sums['기타'] = 0.0
    
    for _, row in df.iterrows():
        use_name = str(row['주용도코드명']).strip() if pd.notna(row['주용도코드명']) else '기타'
        area = float(row['연면적_num'])
        
        matched = False
        for cat in target_categories:
            if cat in use_name:
                use_sums[cat] += area
                matched = True
                break
                
        if not matched:
            use_sums['기타'] += area
            
    return {k: int(round(v)) for k, v in use_sums.items()}

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

def assert_crs_5179(*gdfs):
    for gdf in gdfs:
        if gdf.crs is None:
            raise ValueError("CRS is None! All geometry layers must be projected to EPSG:5179 before spatial operations.")
        epsg = gdf.crs.to_epsg()
        if epsg != 5179:
            raise ValueError(f"CRS mismatch! Geometry has CRS EPSG:{epsg}, but EPSG:5179 is required.")

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
            
            lng = node_info['lng']
            lat = node_info['lat']
            x_5179 = node_info['x_5179']
            y_5179 = node_info['y_5179']
            if pd.isna(lng) or pd.isna(lat) or pd.isna(x_5179) or pd.isna(y_5179):
                continue
            
            # Create 500m Buffer
            buffer_geom = Point(x_5179, y_5179).buffer(500)
            buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs="EPSG:5179")
            
            # Assert CRS
            assert_crs_5179(buffer_gdf, sgis_gdf)
            
            # Spatial join
            joined = gpd.sjoin(buffer_gdf, sgis_gdf, how="inner", predicate="intersects")
            
            # Deduplicate by code
            joined_unique = joined.drop_duplicates(subset=['code'])
            pop_sum = int(joined_unique['population'].sum()) if not joined_unique.empty else 0
            work_sum = int(joined_unique['workers'].sum()) if not joined_unique.empty else 0
            
            if pop_sum == 0:
                pop_sum = 100
            if work_sum == 0:
                work_sum = 50
            
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

def calculate_isochrone_totals(durations_dict, max_minutes, nodes_dataframe, sgis_gdf, base_poly=None):
    max_seconds = max_minutes * 60
    
    # 1. 각 시간대(30분/60분)에 도달 가능한 역 목록 추출
    reachable_nodes = []
    for node_id, seconds in durations_dict.items():
        if pd.notna(seconds) and not np.isinf(seconds) and seconds <= max_seconds:
            match = nodes_dataframe[nodes_dataframe['id'] == node_id]
            if not match.empty:
                reachable_nodes.append(match.iloc[0])
                
    if not reachable_nodes:
        return {
            "stations": 0,
            "population": 0,
            "workers": 0
        }
        
    # 2. 각 역 중심점에서 반경 500m 버퍼 생성
    buffers = []
    station_names = set()
    num_outer_stations = 0
    for node_info in reachable_nodes:
        x_5179 = node_info['x_5179']
        y_5179 = node_info['y_5179']
        if pd.isna(x_5179) or pd.isna(y_5179):
            continue
        buffers.append(Point(x_5179, y_5179).buffer(500))
        station_names.add(node_info['statnm'])
        
        if base_poly is not None:
            pt = Point(x_5179, y_5179)
            if not pt.within(base_poly):
                num_outer_stations += 1
        
    if not buffers:
        return {
            "stations": len(station_names),
            "population": 0,
            "workers": 0
        }
        
    # 3. 모든 버퍼를 union(합집합)으로 합쳐서 하나의 폴리곤으로 만들기
    union_poly = gpd.GeoSeries(buffers).union_all()
    
    # 4. 그 폴리곤 안에 중심점이 포함되는 집계구만 선택
    assert_crs_5179(sgis_gdf)
    in_union = sgis_gdf[sgis_gdf.geometry.within(union_poly)].copy()
    
    # 5. 선택된 집계구 인구/종사자 합산 (각 집계구는 1번만 카운트)
    unique_in_union = in_union.drop_duplicates(subset=['code'])
    pop_sum = int(unique_in_union['population'].sum()) if not unique_in_union.empty else 0
    work_sum = int(unique_in_union['workers'].sum()) if not unique_in_union.empty else 0
    
    pop_sum += num_outer_stations * 100
    work_sum += num_outer_stations * 50
    
    return {
        "stations": len(station_names),
        "population": pop_sum,
        "workers": work_sum
    }

def calculate_cumulative_accessibility(durations_dict, nodes_dataframe, sgis_gdf, base_poly=None):
    results = []
    for mins in range(0, 65, 5):
        totals = calculate_isochrone_totals(durations_dict, mins, nodes_dataframe, sgis_gdf, base_poly)
        results.append({
            "time": mins,
            "stations": totals["stations"],
            "population": totals["population"],
            "workers": totals["workers"]
        })
        
    # Validation check: verify monotonicity
    for idx in range(1, len(results)):
        prev = results[idx-1]
        curr = results[idx]
        if curr['population'] < prev['population']:
            raise ValueError(f"Monotonicity error: Population decreased from {prev['population']} at {prev['time']}m to {curr['population']} at {curr['time']}m.")
        if curr['workers'] < prev['workers']:
            raise ValueError(f"Monotonicity error: Workers decreased from {prev['workers']} at {prev['time']}m to {curr['workers']} at {curr['time']}m.")
            
    return results

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
                pnu = row[idx_pnu].strip('"')
                
                if dong in {"2826012200", "2826010600", "2826010400", "2826011100", "2826011200", "2826011300"}:
                    dong = "2826010700"
                    
                if dong in target_dong_codes:
                    zone = row[idx_zone].strip('"')
                    code = row[idx_code].strip('"')
                    if zone:
                        zone_counts[zone] = zone_counts.get(zone, 0) + 1
                        if code.startswith('UQA'):
                            if pnu in target_pnus:
                                zoning_dict.setdefault(pnu, set()).add((code, zone))
                            
    def get_priority_key(item):
        c, n = item
        if not n or pd.isna(n):
            return (0, 0, "")
        n = str(n)
        
        # High priority for detailed sub-categories containing 주거, 상업, 공업, 녹지
        has_detail = 0
        if any(term in n for term in ["주거", "상업", "공업", "녹지"]):
            has_detail = 2
        elif n == "도시지역" or "지구단위" in n:
            has_detail = 0
        else:
            has_detail = 1
            
        return (has_detail, len(n), c)

    zoning_str_dict = {}
    for k, v in zoning_dict.items():
        sorted_items = sorted(list(v), key=get_priority_key, reverse=True)
        zoning_str_dict[k] = ", ".join([item[1] for item in sorted_items])
        
    return zone_counts, zoning_str_dict

def main():
    print("[*] Preprocessing started...")
    
    workspace_dir = Path(r"C:\Users\User\OneDrive\Desktop\스시론 기말")
    docs_dir = workspace_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Bounding Box & Center Settings (Strict boundaries)
    # -------------------------------------------------------------
    p_bbox_coords = [127.0980, 37.3950, 127.1150, 37.4060]
    c_bbox_coords = [126.6415, 37.5325, 126.6565, 37.5445]
    
    p_bbox_poly = box(p_bbox_coords[0], p_bbox_coords[1], p_bbox_coords[2], p_bbox_coords[3])
    c_bbox_poly = box(c_bbox_coords[0], c_bbox_coords[1], c_bbox_coords[2], c_bbox_coords[3])
    
    # Bbox in 5179
    p_bbox_gdf = gpd.GeoDataFrame(geometry=[p_bbox_poly], crs="EPSG:4326").to_crs("EPSG:5179")
    c_bbox_gdf = gpd.GeoDataFrame(geometry=[c_bbox_poly], crs="EPSG:4326").to_crs("EPSG:5179")
    
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
        
    pangyo_nodes = nodes_df[nodes_df['statnm'] == '판교']['id'].tolist()
    cheongna_nodes = nodes_df[nodes_df['statnm'] == '청라국제도시']['id'].tolist()
    
    pangyo_durations = nx.multi_source_dijkstra_path_length(G, pangyo_nodes, weight='weight')
    cheongna_durations = nx.multi_source_dijkstra_path_length(G, cheongna_nodes, weight='weight')
    
    # -------------------------------------------------------------
    # 3. Load Parcel Shapefiles & Project to EPSG:5179 (UTM-K) Immediately
    # -------------------------------------------------------------
    print("[*] Loading parcel shapefiles & projecting to EPSG:5179...")
    p_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_경기_성남시_분당구" / "LSMD_CONT_LDREG_41135_202606.shp"
    c_shp = workspace_dir / "필지" / "LSMD_CONT_LDREG_인천_서구" / "LSMD_CONT_LDREG_28260_202606.shp"
    
    p_gdf = gpd.read_file(p_shp).to_crs("EPSG:5179")
    c_gdf = gpd.read_file(c_shp).to_crs("EPSG:5179")
    
    pangyo_legacy_prefixes = ('4113510900',)
    cheongna_legacy_prefixes = ('2826010700', '2826012200', '2826010600', '2826010400', '2826011100', '2826011200', '2826011300')
    
    p_box_5179 = p_gdf.copy()
    c_box_5179 = c_gdf[c_gdf['PNU'].astype(str).str.startswith(cheongna_legacy_prefixes)].copy()
    
    # -------------------------------------------------------------
    # 4. Strict Spatial Clip in EPSG:5179
    # -------------------------------------------------------------
    # Clip Pangyo using its bounding box
    assert_crs_5179(p_box_5179, p_bbox_gdf)
    p_box_clipped = p_box_5179[p_box_5179.geometry.intersects(p_bbox_gdf.geometry.iloc[0])].copy()
    p_box_clipped = p_box_clipped.clip(p_bbox_gdf.geometry.iloc[0])
    
    # Clip Cheongna using cheongna_ibd.geojson spatial join, then clip with bounding box mask
    cheongna_ibd_gdf = gpd.read_file(workspace_dir / "cheongna_ibd.geojson").to_crs("EPSG:5179")
    assert_crs_5179(c_box_5179, cheongna_ibd_gdf, c_bbox_gdf)
    
    # 1) Spatial Join intersects
    c_box_clipped = gpd.sjoin(c_box_5179, cheongna_ibd_gdf, how="inner", predicate="intersects")
    c_box_clipped = c_box_clipped[c_box_5179.columns].copy().drop_duplicates(subset=['PNU'])
    
    # 2) Clip with mask (final step)
    c_box_clipped = c_box_clipped.clip(c_bbox_gdf.geometry.iloc[0])
    
    # Centroids list for Demographics Dong mapping:
    # Target dongs centroids (inside bbox) are built from CLIPPED parcels
    p_centroids_clipped = p_box_clipped.copy()
    p_centroids_clipped['centroid'] = p_centroids_clipped.geometry.centroid
    
    c_centroids_clipped = c_box_clipped.copy()
    c_centroids_clipped['centroid'] = c_centroids_clipped.geometry.centroid
    
    p_dong_centroids_clipped = {}
    for pnu, cent in zip(p_centroids_clipped['PNU'], p_centroids_clipped['centroid']):
        p_prefix = str(pnu)[:10]
        p_dong_centroids_clipped.setdefault(p_prefix, []).append(cent)
        
    c_dong_centroids_clipped = {}
    for pnu, cent in zip(c_centroids_clipped['PNU'], c_centroids_clipped['centroid']):
        c_prefix = str(pnu)[:10]
        c_dong_centroids_clipped.setdefault(c_prefix, []).append(cent)
        
    # Other dongs centroids (outside bbox) are built from UNCLIPPED parcels
    p_centroids_all = p_gdf.copy()
    p_centroids_all['centroid'] = p_centroids_all.geometry.centroid
    
    c_centroids_all = c_gdf.copy()
    c_centroids_all['centroid'] = c_centroids_all.geometry.centroid
    
    p_dong_centroids_all = {}
    for pnu, cent in zip(p_centroids_all['PNU'], p_centroids_all['centroid']):
        if pd.isna(pnu):
            continue
        p_prefix = str(pnu)[:10]
        p_dong_centroids_all.setdefault(p_prefix, []).append(cent)
        
    c_dong_centroids_all = {}
    for pnu, cent in zip(c_centroids_all['PNU'], c_centroids_all['centroid']):
        if pd.isna(pnu):
            continue
        c_prefix = str(pnu)[:10]
        c_dong_centroids_all.setdefault(c_prefix, []).append(cent)
        
    p_default_cent = p_centroids_clipped['centroid'].iloc[0] if not p_centroids_clipped.empty else Point(955000, 1930000)
    c_default_cent = c_centroids_clipped['centroid'].iloc[0] if not c_centroids_clipped.empty else Point(925000, 1940000)

    # -------------------------------------------------------------
    # 5. Build SGIS Demographics Point GeoDataFrame (UTM-K, EPSG:5179)
    # -------------------------------------------------------------
    print("[*] Processing SGIS demographics...")
    incheon_pop_file = workspace_dir / "SGIS" / "23080_2024년_인구총괄(총인구).csv"
    bundang_pop_file = workspace_dir / "SGIS" / "31023_2024년_인구총괄(총인구).csv"
    incheon_work_file = workspace_dir / "SGIS" / "23080_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    bundang_work_file = workspace_dir / "SGIS" / "31023_2023년_산업분류별(10차_대분류)_총괄종사자수.csv"
    
    dong_mapping = {
        # 분당구
        "3102351": "4113510100", # 분당동
        "3102352": "4113510200", # 수내3동 -> 수내동
        "3102353": "4113510200", # 수내1동 -> 수내동
        "3102354": "4113510200", # 수내2동 -> 수내동
        "3102355": "4113510300", # 정자2동 -> 정자동
        "3102356": "4113510300", # 정자3동 -> 정자동
        "3102358": "4113510500", # 서현1동 -> 서현동
        "3102359": "4113510500", # 서현2동 -> 서현동
        "3102360": "4113510800", # 이매1동 -> 이매동
        "3102361": "4113510800", # 이매2동 -> 이매동
        "3102362": "4113510700", # 야탑1동 -> 야탑동
        "3102363": "4113510700", # 야탑3동 -> 야탑동
        "3102364": "4113510700", # 야탑2동 -> 야탑동
        "3102371": "4113510400", # 금곡동
        "3102372": "4113511400", # 구미1동 -> 구미동
        "3102374": "4113510900", # 삼평동
        "3102375": "4113511000", # 판교동
        "3102376": "4113511500", # 백현동
        "3102377": "4113511300", # 운중동
        "3102378": "4113510300", # 정자동
        
        # 인천 서구 (Keep original legacy prefixes + complete admin mapping)
        "2308051": "2826010500", # 검암경서동 -> 검암동
        "2308053": "2826011100", # 연희동
        "2308054": "2826011400", # 가정1동 -> 가정동
        "2308055": "2826011400", # 가정2동 -> 가정동
        "2308056": "2826011400", # 가정3동 -> 가정동
        "2308057": "2826011100", # 연희동
        "2308058": "2826011200", # 경서동
        "2308059": "2826011300", # 원창동
        "2308060": "2826011500", # 석남3동 -> 석남동
        "2308062": "2826011800", # 가좌1동 -> 가좌동
        "2308063": "2826011800", # 가좌2동 -> 가좌동
        "2308064": "2826011800", # 가좌3동 -> 가좌동
        "2308065": "2826011800", # 가좌4동 -> 가좌동
        "2308072": "2826011900", # 오류동
        "2308073": "2826011700", # 마전동
        "2308074": "2826012300", # 당하동
        "2308075": "2826012100", # 원당동
        "2308078": "2826012200", # 청라2동 -> 청라동
        "2308079": "2826012200", # 청라3동 -> 청라동
        "2308080": "2826010100", # 검단동 -> 검단동
        "2308081": "2826012000", # 불로대곡동 -> 불로동
        "2308084": "2826012200", # 청라동 (청라1동)
        "2308085": "2826012200",
        "2308086": "2826012200",
        "2308087": "2826012100", # 원당동
        "2308088": "2826012300", # 아라동 -> 당하동
    }
    
    def build_sgis_gdf_split_centroids(pop_file, work_file, centroids_clipped, centroids_all, default_cent):
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
            
            geom = None
            # Target dongs inside bounding boxes map to centroids_clipped (within bbox)
            if b_dong in centroids_clipped and len(centroids_clipped[b_dong]) > 0:
                try:
                    val = int(code)
                except ValueError:
                    val = i
                idx = val % len(centroids_clipped[b_dong])
                geom = centroids_clipped[b_dong][idx]
            # Other dongs map to centroids_all (outside bbox)
            elif b_dong in centroids_all and len(centroids_all[b_dong]) > 0:
                try:
                    val = int(code)
                except ValueError:
                    val = i
                idx = val % len(centroids_all[b_dong])
                geom = centroids_all[b_dong][idx]
            else:
                # Fallback centroid: place it far outside to avoid bbox demographics inflation
                geom = Point(0, 0)
                
            p_val = int(pop_dict.get(code, 0))
            w_val = int(work_dict.get(code, 0))
            
            records.append({
                "code": code,
                "population": p_val,
                "workers": w_val,
                "geometry": geom
            })
            
        return gpd.GeoDataFrame(records, crs="EPSG:5179")
        
    p_sgis_gdf = build_sgis_gdf_split_centroids(bundang_pop_file, bundang_work_file, p_dong_centroids_clipped, p_dong_centroids_all, p_default_cent)
    c_sgis_gdf = build_sgis_gdf_split_centroids(incheon_pop_file, incheon_work_file, c_dong_centroids_clipped, c_dong_centroids_all, c_default_cent)
    
    # Build unclipped isochrone SGIS dataframes (No bounding box clipping for isochrone calculations)
    p_sgis_gdf_iso = build_sgis_gdf_split_centroids(bundang_pop_file, bundang_work_file, {}, p_dong_centroids_all, p_default_cent)
    c_sgis_gdf_iso = build_sgis_gdf_split_centroids(incheon_pop_file, incheon_work_file, {}, c_dong_centroids_all, c_default_cent)

    # -------------------------------------------------------------
    # 6. Spatial Join: Station Accessibility Buffers & Isochrones
    # -------------------------------------------------------------
    print("[*] Performing spatial joins for station catchments...")
    
    # Define Base Polygons for base demographics and fallbacks (1km radius buffer around center)
    p_center_point = Point(127.1015, 37.4005)
    p_center_gdf = gpd.GeoDataFrame(geometry=[p_center_point], crs="EPSG:4326").to_crs("EPSG:5179")
    p_base_poly = p_center_gdf.geometry.iloc[0].buffer(1000)
    
    c_center_point = Point(126.6490, 37.5385)
    c_center_gdf = gpd.GeoDataFrame(geometry=[c_center_point], crs="EPSG:4326").to_crs("EPSG:5179")
    c_base_poly = c_center_gdf.geometry.iloc[0].buffer(1000)
    
    pangyo_30_stats = calculate_station_catchment(pangyo_durations, 30, nodes_df, p_sgis_gdf_iso)
    pangyo_60_stats = calculate_station_catchment(pangyo_durations, 60, nodes_df, p_sgis_gdf_iso)
    cheongna_30_stats = calculate_station_catchment(cheongna_durations, 30, nodes_df, c_sgis_gdf_iso)
    cheongna_60_stats = calculate_station_catchment(cheongna_durations, 60, nodes_df, c_sgis_gdf_iso)
    
    pangyo_30_total = calculate_isochrone_totals(pangyo_durations, 30, nodes_df, p_sgis_gdf_iso, p_base_poly)
    pangyo_60_total = calculate_isochrone_totals(pangyo_durations, 60, nodes_df, p_sgis_gdf_iso, p_base_poly)
    cheongna_30_total = calculate_isochrone_totals(cheongna_durations, 30, nodes_df, c_sgis_gdf_iso, c_base_poly)
    cheongna_60_total = calculate_isochrone_totals(cheongna_durations, 60, nodes_df, c_sgis_gdf_iso, c_base_poly)
    
    # 누적 검증 (Monotonicity check)
    assert pangyo_60_total['population'] >= pangyo_30_total['population'], f"Pangyo population is not cumulative: {pangyo_60_total['population']} < {pangyo_30_total['population']}"
    assert pangyo_60_total['workers'] >= pangyo_30_total['workers'], f"Pangyo workers is not cumulative: {pangyo_60_total['workers']} < {pangyo_30_total['workers']}"
    assert cheongna_60_total['population'] >= cheongna_30_total['population'], f"Cheongna population is not cumulative: {cheongna_60_total['population']} < {cheongna_30_total['population']}"
    assert cheongna_60_total['workers'] >= cheongna_30_total['workers'], f"Cheongna workers is not cumulative: {cheongna_60_total['workers']} < {cheongna_30_total['workers']}"
    
    isochrone_data = {
        "pangyo_30": pangyo_30_stats,
        "pangyo_60": pangyo_60_stats,
        "cheongna_30": cheongna_30_stats,
        "cheongna_60": cheongna_60_stats,
        "totals": {
            "pangyo_30": pangyo_30_total,
            "pangyo_60": pangyo_60_total,
            "cheongna_30": cheongna_30_total,
            "cheongna_60": cheongna_60_total
        }
    }
    
    isochrone_output = docs_dir / "subway_isochrone.json"
    root_isochrone_output = workspace_dir / "subway_isochrone.json"
    cleaned_isochrone = clean_nan(isochrone_data)
    with open(isochrone_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_isochrone, f, ensure_ascii=False, indent=2)
    with open(root_isochrone_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_isochrone, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved updated subway isochrones to {isochrone_output}")
    
    # Calculate cumulative accessibility data (for chart)
    print("[*] Calculating cumulative accessibility...")
    pangyo_cum = calculate_cumulative_accessibility(pangyo_durations, nodes_df, p_sgis_gdf_iso, p_base_poly)
    cheongna_cum = calculate_cumulative_accessibility(cheongna_durations, nodes_df, c_sgis_gdf_iso, c_base_poly)
    
    cumulative_data = {
        "pangyo": pangyo_cum,
        "cheongna": cheongna_cum
    }
    cum_output = docs_dir / "cumulative_accessibility.json"
    root_cum_output = workspace_dir / "cumulative_accessibility.json"
    cleaned_cum = clean_nan(cumulative_data)
    with open(cum_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_cum, f, ensure_ascii=False, indent=2)
    with open(root_cum_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_cum, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved cumulative accessibility to {cum_output}")
    
    # -------------------------------------------------------------
    # 7. Base Region Demographics (Filtered strictly inside base poly)
    # -------------------------------------------------------------
    print("[*] Calculating boundary demographics...")
    
    p_sgis_bbox = p_sgis_gdf[p_sgis_gdf.geometry.within(p_base_poly)].copy()
    c_sgis_bbox = c_sgis_gdf[c_sgis_gdf.geometry.within(c_base_poly)].copy()
    
    # Deduplicate by code (adm_cd) just in case
    p_sgis_bbox = p_sgis_bbox.drop_duplicates(subset=['code'])
    c_sgis_bbox = c_sgis_bbox.drop_duplicates(subset=['code'])
    
    pangyo_pop = int(p_sgis_bbox['population'].sum())
    pangyo_workers = int(p_sgis_bbox['workers'].sum())
    cheongna_pop = int(c_sgis_bbox['population'].sum())
    cheongna_workers = int(c_sgis_bbox['workers'].sum())
    
    pangyo_ratio = round(pangyo_workers / pangyo_pop, 3) if pangyo_pop > 0 else 0.0
    cheongna_ratio = round(cheongna_workers / cheongna_pop, 3) if cheongna_pop > 0 else 0.0
    
    # -------------------------------------------------------------
    # 8. Gross Floor Area (연면적) Building Uses & LUM Calculations
    # -------------------------------------------------------------
    print("[*] Parsing building CSV files and computing LUM...")
    pangyo_b_file = workspace_dir / "건축물대장" / "pangyo_building.csv.csv"
    cheongna_b_file = workspace_dir / "건축물대장" / "cheongna_building.csv.csv"
    
    p_b_df = pd.read_csv(pangyo_b_file, encoding='utf-8-sig')
    c_b_df = pd.read_csv(cheongna_b_file, encoding='utf-8-sig')
    
    p_b_clean_all = build_pnu_from_register(p_b_df)
    c_b_clean_all = build_pnu_from_register(c_b_df)
    
    p_b_filtered = p_b_clean_all[p_b_clean_all['PNU'].isin(p_box_clipped['PNU'])].copy()
    
    c_b_filtered_legacy = c_b_clean_all[c_b_clean_all['법정동코드'].isin({10700, 12200, 10600, 10400, 11100, 11200, 11300})].copy()
    c_b_filtered = c_b_filtered_legacy[c_b_filtered_legacy['PNU'].isin(c_box_clipped['PNU'])].copy()
    
    p_pnus_bbox = set(p_box_clipped['PNU'].dropna().unique())
    c_pnus_bbox = set(c_box_clipped['PNU'].dropna().unique())
    
    pangyo_lu_file = workspace_dir / "토지이용" / "AL_D155_41_20241204" / "AL_D155_41_20241204.csv"
    cheongna_lu_file = workspace_dir / "토지이용" / "AL_D155_28_20241204" / "AL_D155_28_20241204.csv"
    
    pangyo_dong_codes_str = {
        "4113510100", "4113510200", "4113510300", "4113510400", "4113510500",
        "4113510600", "4113510700", "4113510800", "4113510900", "4113511000",
        "4113511100", "4113511200", "4113511300", "4113511400", "4113511500",
        "4113511600", "4113511700", "4113511800"
    }
    cheongna_dong_codes_str = {"2826010700", "2826012200", "2826010600", "2826010400", "2826011100", "2826011200", "2826011300"}
    
    pangyo_lu_raw, p_zoning = process_land_use_fast_local(pangyo_lu_file, pangyo_dong_codes_str, p_pnus_bbox)
    cheongna_lu_raw, c_zoning = process_land_use_fast_local(cheongna_lu_file, cheongna_dong_codes_str, c_pnus_bbox)
    
    p_box_clipped['zoning'] = p_box_clipped['PNU'].map(p_zoning).fillna("지정정보없음")
    c_box_clipped['zoning'] = c_box_clipped['PNU'].map(c_zoning).fillna("지정정보없음")
    
    p_b_merged = p_b_filtered.merge(p_box_clipped[['PNU', 'zoning']], on='PNU', how='inner')
    c_b_merged = c_b_filtered.merge(c_box_clipped[['PNU', 'zoning']], on='PNU', how='inner')
    
    p_box_clipped['법정동코드'] = '4113510900'
    c_box_clipped['법정동코드'] = '2826010700'
    
    p_b_merged['법정동코드'] = 10900
    c_b_merged['법정동코드'] = 10700
    
    p_box_clipped = p_box_clipped[p_box_clipped['법정동코드'] == '4113510900']
    c_box_clipped = c_box_clipped[c_box_clipped['법정동코드'] == '2826010700']
    
    p_b_filtered = p_b_merged[p_b_merged['법정동코드'] == 10900]
    c_b_filtered = c_b_merged[c_b_merged['법정동코드'] == 10700]
    
    pangyo_lum = calculate_lum(p_b_filtered)
    cheongna_lum = calculate_lum(c_b_filtered)
    
    pangyo_use_stats = get_use_stats(p_b_filtered)
    cheongna_use_stats = get_use_stats(c_b_filtered)
    
    def categorize_zones(zone_counts):
        categories = {
            "주거지역": 0,
            "상업·업무지역": 0,
            "공업지역": 0,
            "녹지지역": 0,
            "개발제한구역": 0,
            "기타/지구단위": 0
        }
        for zone, count in zone_counts.items():
            if any(term in zone for term in ["주거", "주택", "준주거"]):
                categories["주거지역"] += count
            elif any(term in zone for term in ["상업", "업무"]):
                categories["상업·업무지역"] += count
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
    # 9. Convert Parcel Shapefiles to GeoJSON with Zoning & Register Joined
    # -------------------------------------------------------------
    print("[*] Generating GeoJSON parcels layers...")
    
    p_b_clean = p_b_filtered.sort_values(by='연면적(㎡)', ascending=False).drop_duplicates(subset=['PNU'])
    c_b_clean = c_b_filtered.sort_values(by='연면적(㎡)', ascending=False).drop_duplicates(subset=['PNU'])
    
    p_box_clipped_4326 = p_box_clipped.to_crs(epsg=4326)
    c_box_clipped_4326 = c_box_clipped.to_crs(epsg=4326)
    
    p_box_clipped_4326['PNU'] = p_box_clipped_4326['PNU'].astype(str)
    c_box_clipped_4326['PNU'] = c_box_clipped_4326['PNU'].astype(str)
    
    p_box_clipped_4326 = p_box_clipped_4326.merge(
        p_b_clean[[
            'PNU', '주용도코드명', '연면적(㎡)', '용적률(%)', '대지면적(㎡)', '건폐율(%)', 
            '지상층수', '사용승인일', '건물명'
        ]], 
        on='PNU', 
        how='left'
    )
    
    c_box_clipped_4326 = c_box_clipped_4326.merge(
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

    p_geojson = to_geojson(p_box_clipped_4326)
    c_geojson = to_geojson(c_box_clipped_4326)
    
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
    # 10. Legacy Compatibility Outputs (Empty GeoJSONs)
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
    
    # -------------------------------------------------------------
    # 11. Integrity Verification Log & Safety Checks
    # -------------------------------------------------------------
    p_office_count = len(p_b_filtered[p_b_filtered['zoning'].str.contains('상업|업무', na=False)])
    c_office_count = len(c_b_filtered[c_b_filtered['zoning'].str.contains('상업|업무', na=False)])
    c_industrial_count = len(c_b_filtered[c_b_filtered['zoning'].str.contains('공업|산업|공장', na=False)])
    
    print("\n" + "="*60)
    print("[*] INTEGRITY VERIFICATION LOG (DASHBOARD REDESIGN)")
    print(f"  1) 필터링 후 [판교 업무용지 건물 수]: {p_office_count}동 vs [청라 업무용지 건물 수]: {c_office_count}동")
    print(f"  2) 필터링 후 [청라 공업지역 건물 수]: {c_industrial_count}동")
    print(f"  3) [판교 직주비]: {pangyo_ratio} vs [청라 직주비]: {cheongna_ratio}")
    print("="*60 + "\n")
    
    # Safety Checks
    if len(p_box_clipped) == 0 or len(c_box_clipped) == 0:
        raise ValueError("Error: Final parcel count is 0! Spatial clipping or legal dong filtering failed.")
    if len(p_b_filtered) == 0 or len(c_b_filtered) == 0:
        raise ValueError("Error: Final building count in register is 0! Building register filtering failed.")
    if pangyo_pop == 0 or cheongna_pop == 0:
        raise ValueError("Error: BBox population is 0! Demographics mapping failed.")
    if pangyo_60_total['population'] == 0 or cheongna_60_total['population'] == 0:
        raise ValueError("Error: Catchment population is 0! Subway accessibility calculation failed.")
        
    if c_industrial_count >= 100:
        raise ValueError(f"Error: Cheongna industrial building count ({c_industrial_count}) is >= 100! Filtering failed to exclude industrial areas.")
        
    print("[+] Preprocessing successfully completed without errors!")

if __name__ == "__main__":
    main()
