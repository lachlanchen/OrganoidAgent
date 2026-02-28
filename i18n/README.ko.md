[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/figs/banner.png" alt="LazyingArt banner" />
</p>

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Status](https://img.shields.io/badge/Status-Active-success)
![Data](https://img.shields.io/badge/Data-Local%20first-4c956c)
![Preview](https://img.shields.io/badge/Preview-Multi--format-f4a261)

OrganoidAgent는 오가노이드 데이터셋을 로컬에서 탐색하고 미리보기할 수 있도록 만든 경량 Tornado 백엔드 + Progressive Web App(PWA) 프런트엔드입니다. 테이블, 현미경 이미지(TIFF 포함), 아카이브, gzip 텍스트 파일, AnnData `.h5ad` 분석 객체를 실용적으로 파일 형식에 맞춰 미리보기할 수 있습니다.

## 개요 🔭

핵심 앱은 최소 설정으로 대화형 데이터셋 탐색이 가능하도록 설계되었습니다.

- `app.py`의 백엔드 API 및 미리보기 엔진
- `web/`의 PWA 프런트엔드
- `scripts/`의 다운로드 헬퍼
- `datasets/`의 로컬 데이터셋 작업공간(git-ignored)

이 저장소에는 인접한 연구/유틸리티 작업공간(`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, `papers` 서브모듈)도 포함되어 있습니다. 이 README에서 설명하는 기본 실행 대상은 최상위 `OrganoidAgent` 앱입니다.

## 주요 기능 ✨

- 크기 및 파일 수 요약을 포함한 로컬 데이터셋 인덱싱
- 추론된 파일 종류와 함께 데이터셋 파일 재귀 목록화
- CSV/TSV/XLS/XLSX 테이블 미리보기 지원
- TIFF/JPG/PNG 이미지 미리보기 지원
- 임베딩/PCA 산점도 미리보기 생성을 포함한 `.h5ad` 요약 지원
- ZIP/TAR/TGZ 아카이브 목록 + 첫 이미지 미리보기 시도 지원
- `.gz` 텍스트 앞부분 줄 미리보기 지원
- 대용량 패키지 데이터셋용 아카이브 추출 엔드포인트
- Markdown 기반 데이터셋 메타데이터 카드 렌더링
- 서비스 워커 및 매니페스트를 갖춘 PWA 프런트엔드
- `datasets/` 하위로 파일 접근을 제한하는 기본 경로 정제(`safe_dataset_path`)

### 한눈에 보기

| 영역 | 제공 내용 |
|---|---|
| 데이터셋 탐색 | 파일 수 및 크기 요약을 포함한 디렉터리 단위 데이터셋 목록 |
| 파일 탐색 | 재귀 목록 및 종류 추론(`image`, `table`, `analysis`, `archive` 등) |
| 풍부한 미리보기 | 테이블, TIFF/이미지, gzip 텍스트 스니펫, 아카이브 내용, AnnData 요약 |
| 분석 시각화 | `obsm` 임베딩 또는 PCA 대체 경로 기반 `.h5ad` 산점도 미리보기 |
| 패키징 지원 | 대형 압축 번들용 아카이브 목록 + 추출 엔드포인트 |
| 웹 UX | 오프라인 친화적 서비스 워커 자산을 갖춘 설치형 PWA |

## 프로젝트 구조 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # 다운로드된 데이터 및 미리보기 캐시 (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # 서브모듈: prompt-is-all-you-need
├─ i18n/                          # 다국어 README 파일용(현재 존재)
├─ BioAgent/                      # 관련 있지만 별도 앱
├─ BioAgentUtils/                 # 관련 학습/데이터 유틸리티
├─ references/
├─ results/
└─ vendor/                        # 외부 서브모듈 (copilot-sdk, paper-agent, codex)
```

## 사전 요구사항 ✅

- Python `3.10+`
- 권장 환경 관리자: `conda` 또는 `venv`

소스 기준으로 추론한 필수/선택 Python 패키지:

| 패키지 | 역할 |
|---|---|
| `tornado` | 서버 시작에 필수 |
| `pandas` | 선택: 테이블 미리보기 지원 |
| `anndata`, `numpy` | 선택: `.h5ad` 미리보기 및 분석 플로팅 |
| `Pillow` | 선택: 이미지 렌더링 및 생성 미리보기 |
| `tifffile` | 선택: TIFF 미리보기 지원 |
| `requests` | 선택: 데이터셋 다운로드 스크립트 |
| `kaggle` | 선택: 약물 스크리닝 스크립트의 Kaggle 다운로드 |

가정 참고: 현재 최상위 앱에는 루트 `requirements.txt`, `pyproject.toml`, `environment.yml`이 없습니다.

## 설치 ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## 사용법 🚀

### 빠른 시작

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # deps가 이미 있다면 선택 사항
python app.py --port 8080
```

`http://localhost:8080`을 엽니다.

### API 스모크 테스트

```bash
curl http://localhost:8080/api/datasets
```

### 데이터 다운로드(선택 사항)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

다운로드된 데이터는 `datasets/`(git-ignored)에 저장됩니다.

## API 엔드포인트 🌐

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | 요약 통계와 함께 데이터셋 목록 반환 |
| `GET` | `/api/datasets/{name}` | 단일 데이터셋의 파일 목록 반환 |
| `GET` | `/api/datasets/{name}/metadata` | Markdown 메타데이터 카드 반환 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 카테고리 기반 파일 목록 반환 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 파일 형식 인지 미리보기 payload 반환 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 아카이브를 같은 위치의 `_extracted` 폴더로 추출 |
| `GET` | `/files/<path>` | 원본 데이터셋 파일 제공 |
| `GET` | `/previews/<path>` | 생성된 미리보기 자산 제공 |

예시 미리보기 호출:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 구성 🧩

현재 런타임 구성은 의도적으로 단순합니다.

- 서버 포트: `app.py`의 `--port` 인자(기본값 `8080`)
- 데이터 디렉터리: 저장소 루트 기준 상대 경로 `datasets/`로 고정
- 미리보기 캐시: `datasets/.cache/previews`
- 메타데이터 매핑: `app.py`의 `DATASET_METADATA` 딕셔너리
- 다운로더용 GitHub API 토큰(선택): `GITHUB_TOKEN` 환경 변수 또는 `--github-token`

가정 참고: 설정 가능한 데이터셋 루트나 프로덕션 서버 설정은 아직 최상위 구성 파일로 노출되어 있지 않습니다.

## 예시 🧪

### 카테고리별 파일 탐색

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### 아카이브 추출

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 선택적 다운로드 모드 실행

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 개발 노트 🛠️

- 백엔드는 `web/`에서 프런트엔드 정적 자산을 제공합니다.
- 서비스 워커와 매니페스트는 `web/sw.js`, `web/manifest.json`에 있습니다.
- 파일 형식 라우팅 및 미리보기는 `app.py`에 구현되어 있습니다.
- 수동 검증(현재 프로젝트 가이드): PWA가 `http://localhost:8080`에서 로드됨
- 수동 검증(현재 프로젝트 가이드): `/api/datasets`가 JSON을 반환함
- 수동 검증(현재 프로젝트 가이드): CSV/XLSX/이미지/아카이브 미리보기가 렌더링됨

## 문제 해결 🩺

- 미리보기 라이브러리 `ModuleNotFoundError`: 누락 패키지(`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`)를 설치하세요.
- 데이터셋 목록이 비어 있음: 데이터가 `datasets/` 하위에 있고 디렉터리가 점(`.`) 접두어가 아닌지 확인하세요.
- `.h5ad` 미리보기 산점도 이미지가 없음: `anndata`, `numpy`, `Pillow` 설치 여부를 확인하세요.
- 대형 아카이브 미리보기/추출 문제: 추출 엔드포인트를 사용하고 추출된 파일을 직접 확인하세요.
- GitHub 다운로더 rate limit 오류: 환경 변수 또는 CLI 플래그로 `GITHUB_TOKEN`을 제공하세요.
- Kaggle 다운로드가 동작하지 않음: `kaggle`을 설치하고 `~/.kaggle/kaggle.json` 자격 증명을 설정하세요.

## 로드맵 🧭

잠재적 다음 개선 사항(이 루트 앱에는 아직 완전히 구현되지 않음):

- 루트 의존성 매니페스트 추가(`requirements.txt` 또는 `pyproject.toml`)
- API 핸들러와 미리보기 함수 자동 테스트 추가
- 데이터셋 루트 및 캐시 설정을 구성 가능하게 개선
- 명시적 프로덕션 실행 프로필 추가(비디버그, reverse proxy 가이드)
- `i18n/` 다국어 문서 확장

## 기여하기 🤝

기여를 환영합니다. 실용적인 워크플로:

1. 포크 후 범위를 좁힌 브랜치를 생성합니다.
2. 변경 범위를 하나의 논리적 영역으로 유지합니다.
3. 앱 시작과 주요 엔드포인트를 수동으로 검증합니다.
4. 요약, 실행 명령, UI 변경 시 스크린샷을 포함해 PR을 엽니다.

이 저장소의 로컬 스타일 규칙:

- Python: 4칸 들여쓰기, snake_case 함수/파일, CapWords 클래스
- 이 앱의 프런트엔드 로직은 `web/app.js`에 유지(불필요한 프레임워크 재작성 지양)
- 주석은 간결하게, 비자명한 로직에만 작성

## 프로젝트 레이아웃(정식 요약) 📌

- `app.py`: Tornado 서버 및 API 라우트
- `web/`: PWA 자산
- `scripts/`: 데이터셋 다운로드 헬퍼
- `datasets/`: 로컬 데이터 저장소
- `papers/`: 참고 자료 서브모듈

## 라이선스 📄

현재 저장소 루트에는 최상위 프로젝트 `LICENSE` 파일이 없습니다.

가정 참고: 루트 라이선스가 추가되기 전까지, 최상위 OrganoidAgent 코드베이스의 재사용/재배포 조건은 명시되지 않은 상태로 간주하세요.

## 후원 및 기부 ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
