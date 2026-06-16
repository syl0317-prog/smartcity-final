# 수도권 지하철 네트워크 데이터셋

QGIS 에서 손으로 편집한 지하철 역/노선/환승 shape 와 노선별 배차간격을 받아,
**일련번호 노드 + (fromNode, toNode, timeFT, timeTF) 링크** 형태의 그래프로 변환하는 한 세트.

- 데이터 좌표계: EPSG:5179 (Korea 2000 / Unified CS, meter) + WGS84 (lng/lat) 동시 제공
- 시점: 2026-05-05 export
- 포함 범위: 1974 1호선 ~ 2032 GTX-C 까지 (`opening.tsv` 참고)
- **보행망은 포함하지 않음.** node = station 뿐, link = subway / transfer 뿐.
  보행망 결합은 받은 쪽에서 별도로 처리.

---

## 폴더 구조

```
subway_network/
├── README.md                    이 파일
├── opening.tsv                  노선 개통 시점 리스트 (cutoff 결정)
├── line_waits.parquet           노선별 배차간격 (초)
├── edit/                        QGIS 편집 원본 (shp + qml + cpg)
│   ├── stations.shp             역 (Point, EPSG:5179)
│   ├── lines.shp                노선 segment (LineString, 양 끝점이 stations 와 일치)
│   ├── transfer.shp             환승 segment (LineString, 양 끝점이 stations 와 일치)
│   ├── 지도.qgz                  QGIS 프로젝트 파일
│   └── *.qml                    QGIS symbology
├── make_network.ipynb           edit + line_waits → network 그래프
└── network/                     (자동 생성) 분석용 출력 — parquet + tsv 두 형식
    ├── nodes.parquet            일련번호 노드 (geopandas/pyarrow 권장)
    ├── nodes.tsv                같은 내용 tsv (geometry → WKT, UTF-8)
    ├── links.parquet            (fromNode, toNode, timeFT, timeTF) 링크
    └── links.tsv                같은 내용 tsv
```

`tsv` 는 geometry 컬럼을 `geometry_wkt` (WKT 문자열) 로 직렬화. excel/R/다른 언어/스크립트 등에서 읽기 편함. parquet 과 행/컬럼 동일 (geometry → geometry_wkt 로 이름만 변경).

---

## 데이터 스키마

### `network/nodes.parquet`

각 station 이 하나의 노드. 환승역은 노선 수만큼 노드가 있다 (예: 서울역은 1호선/4호선/공항철도/경의중앙선 → 4 노드).

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `id` | int32 | 일련번호 (0..N-1). links.fromNode / toNode 가 이 값을 가리킴. |
| `linenm` | string | 노선 이름 (예: `서울1호선`, `수인선`, `GTX_A`) |
| `statnm` | string | 역 이름 |
| `x_5179`, `y_5179` | float64 | EPSG:5179 좌표 (meter) |
| `lng`, `lat` | float64 | WGS84 (EPSG:4326) |
| `begin` | string `YYYY-MM-DD` | 운영 시작일 (= 노선 개통일 또는 역 신설일) |
| `effective_begin` | string `YYYY-MM-DD` 또는 `''` | 인프라는 완공됐지만 미운영인 경우의 실제 운영 시작일. 비어있으면 begin 과 동일. |
| `geometry` | Point | EPSG:5179 Point |

### `network/links.parquet`

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `id` | int32 | 일련번호 |
| `fromNode`, `toNode` | int32 | nodes.id 참조 |
| `timeFT` | float64 | from → to 통과시간 (초). transfer 의 경우 대기시간 포함. |
| `timeTF` | float64 | to → from 통과시간 (초). 비대칭. |
| `kind` | string | `subway` (lines) 또는 `transfer` |
| `begin` | string `YYYY-MM-DD` | 이 링크의 운영 시작일 |
| `linenm_from` | string | from 쪽 노선 |
| `linenm_to` | string | to 쪽 노선 (subway 는 from 과 동일, transfer 는 다를 수 있음) |
| `length_m` | float64 | LineString 길이 (m) — 참고용 |
| `geometry` | LineString | EPSG:5179 |

#### 시간 정책

**subway (lines) 링크** — 양방향 동일:
```
timeFT = timeTF = lines.time   # 정차/통과 다 포함된 셀 단위 운행 시간
```

**transfer 링크** — 비대칭. 환승 시 도착 노선의 배차간격을 더해 평균 대기시간이 자연스럽게 누적되게 만든다:
```
timeFT = transfer.time + line_waits[linenm_to]
timeTF = transfer.time + line_waits[linenm_from]
```

예: 서울1호선 (배차 150초) ↔ 경의중앙선 (배차 390초), 보행 190초:
- 1호선 → 중앙선 = 190 + 390 = **580초** (중앙선 기다리는 시간 포함)
- 중앙선 → 1호선 = 190 + 150 = **340초**

`line_waits.parquet` 에 노선이 없으면 0초로 처리하고 경고 출력.

### `line_waits.parquet`

| 컬럼 | 의미 |
|---|---|
| `linenm` | 노선 이름 |
| `waittm` | 배차간격 (초). 평균 대기시간 = waittm / 2 가 일반적이지만 본 데이터는 **waittm 자체를 대기 추가값으로 사용** (피크/비피크 평균치 + 보수적 가정). |

### `opening.tsv`

| 컬럼 | 의미 |
|---|---|
| `date` | `YYYY-MM-DD` |
| `desc` | 해당 시점의 이벤트 (`1호선 개통`, `GTX_A 개통`, `현재` 등) |

`make_network.ipynb` 가 `opening.tsv` 의 `max(date)` 를 cutoff 로 사용해, 그보다 미래에 `begin` 이 잡힌 stations/lines 를 제외한다.

---

## 사용법

### 0. 사전 준비

```powershell
pip install geopandas pyogrio pandas pyarrow shapely jupyterlab
```

```powershell
cd subway_network
jupyter lab
```

### 1. make_network.ipynb 실행

cell 1~14 순서대로 실행:

- **cell 2~5** — stations / lines / transfer shp 로드 + 정규화.
- **cell 3** — `opening.tsv` 의 max date 를 cutoff 로 잡음. 없으면 `2099-12-31` fallback.
- **cell 6** — transfer 의 양 끝점이 stations 의 점과 1m 이내 매칭 안 되면 제거.
- **cell 7** — 누락된 환승 검출 (동일 statnm + 다른 linenm 페어 중 1km 이내인데 transfer 에 없는 것).
- **cell 8** — `EXTRA_TRANSFERS` 에 `(statnm, time_seconds)` 튜플로 누락 환승 수동 추가. **기본은 빈 리스트** — cell 7 출력 보고 채워 넣는다.

예시:
```python
EXTRA_TRANSFERS = [
    ('곡산', 10),    # 경의중앙선 <-> 서해선
    ('백마', 10),
    ('일산', 10),
    ('풍산', 10),
    ('구성', 300),   # 분당선 <-> GTX_A (109m 떨어짐, 300초)
]
```

같은 statnm 이 정확히 2개일 때만 동작 (그 둘을 잇는 직선 LineString 생성). 3개 이상이면 `edit/transfer.shp` 에 직접 그려야 한다.

- **cell 9** — stations 에 일련번호 `id` (0..N-1) 부여.
- **cell 10** — line_waits 로드. 누락 노선 경고.
- **cell 11** — lines / transfer 의 양 끝점을 nodes.id 와 매칭 (sjoin_nearest, 1m).
- **cell 12** — link 테이블 생성 (subway / transfer 통합).
- **cell 13** — node 테이블 생성 (5179 좌표 + WGS84).
- **cell 14** — 저장 + 검증 (link 의 fromNode/toNode 가 nodes 범위 안인지).

### 2. 결과 확인

```python
import geopandas as gpd
nodes = gpd.read_parquet('network/nodes.parquet')
links = gpd.read_parquet('network/links.parquet')
print(nodes.shape, links.shape)
print(links['kind'].value_counts())
```

QGIS 에서 `edit/지도.qgz` 로 편집 view, `network/links.parquet` 로 가공된 그래프 확인 가능.

---

## 그래프 분석 활용

### parquet 로드 (geometry 포함, 권장)

```python
import geopandas as gpd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

nodes = gpd.read_parquet('network/nodes.parquet')
links = gpd.read_parquet('network/links.parquet')
V = len(nodes)

# directed CSR (timeFT, timeTF 둘 다 행으로 만들어 양방향 directed 로)
u = links['fromNode'].to_numpy()
v = links['toNode'].to_numpy()
src = np.concatenate([u, v])
dst = np.concatenate([v, u])
cost = np.concatenate([links['timeFT'].to_numpy(), links['timeTF'].to_numpy()]).astype(np.float32)
A = csr_matrix((cost, (src, dst)), shape=(V, V))

# 서울역 (index 찾아서) 출발 SSSP
seoul = nodes.query("statnm == '서울' and linenm == '서울1호선'").iloc[0]
sol = dijkstra(A, indices=int(seoul['id']))
print('30분 이내 도달:', int(((sol > 0) & (sol < 1800)).sum()), '노드')
```

### tsv 로드 (geopandas 없이, 다른 언어/도구용)

geometry 가 필요 없으면 그냥 pandas / csv.reader / R / excel 로 읽으면 됨:

```python
import pandas as pd
nodes = pd.read_csv('network/nodes.tsv', sep='\t')
links = pd.read_csv('network/links.tsv', sep='\t')
# geometry 가 필요하면 WKT → shapely
from shapely import wkt
nodes['geometry'] = nodes['geometry_wkt'].apply(wkt.loads)
```

R:
```r
nodes <- read.delim("network/nodes.tsv", encoding = "UTF-8")
links <- read.delim("network/links.tsv", encoding = "UTF-8")
```

excel: `nodes.tsv` 더블클릭 → 탭 구분으로 자동 인식.

### 시점 분석 (`begin` / `effective_begin`)

특정 시점 `T` (예: `2026-05-04`) 의 운영 네트워크만 분석:

```python
T = '2026-05-04'

# 활성 노드: effective_begin (있으면) 또는 begin 이 T 이하인 것
nodes_eff = nodes['effective_begin'].where(nodes['effective_begin'] != '', nodes['begin'])
active_nodes = nodes[nodes_eff <= T]
active_ids = set(active_nodes['id'])

# 활성 링크: begin 이 T 이하 + 양 끝 노드 모두 active
active_links = links[
    (links['begin'] <= T)
    & links['fromNode'].isin(active_ids)
    & links['toNode'].isin(active_ids)
]
```

---

## 알려진 함정

1. **shp 컬럼명은 10자 제한** — `effective_begin` 이 `eff_begin` 으로 줄어든다. 노트북이 자동 복원.
2. **shp dbf 인코딩은 UTF-8** — `*.cpg` 가 `UTF-8` 이어야 함. cp949 로 떨어지면 한글 깨짐. (QGIS 는 cpg 따라 정상.)
3. **endpoint 가 station 과 1m 안 떨어져야 한다** — lines/transfer 를 QGIS 에서 그릴 때 snap 켜고 station 에 정확히 붙여야 한다. cell 6 / 11 에서 매칭 실패하면 drop 됨.
4. **`time` 단위는 초** — lines.time, transfer.time, line_waits.waittm 모두 초.
5. **`begin` 컬럼은 문자열** (`YYYY-MM-DD`) — 비교는 문자열 lexicographic 으로 잘 작동.
6. **link 가 directed 가 아니라 한 행에 양방향 cost 둘 다** — 분석 시 양방향 directed 그래프로 풀려면 `timeFT` 와 `timeTF` 를 두 행으로 펼쳐야 함 (위 SSSP 예제 참고).

---

## 데이터 변경 시

QGIS 에서 `edit/*.shp` 직접 편집 → 노트북 다시 실행 → `network/` 자동 갱신.

새 노선/시점 추가 시:
1. `edit/stations.shp`, `edit/lines.shp`, `edit/transfer.shp` 편집.
2. `opening.tsv` 에 한 줄 추가 (cutoff 동기화).
3. `line_waits.parquet` 에 새 노선의 배차간격 추가 (없으면 0초로 처리 + 경고).
4. 노트북 재실행.

---

## 라이선스 / 출처

OSM, KRRI 보고서, 각 사업자 보도자료 등을 참고하여 직접 손으로 그린 데이터.
비영리 연구/시각화 용도로 자유 활용 가능. 상업적 사용 시 개별 확인 필요.
