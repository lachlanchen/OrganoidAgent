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

OrganoidAgent là một backend Tornado + frontend Progressive Web App (PWA) gọn nhẹ để duyệt và xem trước bộ dữ liệu organoid trên máy cục bộ. Ứng dụng hỗ trợ xem trước thực tiễn theo từng loại tệp cho bảng dữ liệu, ảnh hiển vi (bao gồm TIFF), tệp nén, tệp văn bản gzip và đối tượng phân tích AnnData `.h5ad`.

## Tổng quan 🔭

Ứng dụng cốt lõi được thiết kế để khám phá bộ dữ liệu tương tác với thiết lập tối thiểu:

- Backend API và engine xem trước trong `app.py`
- Frontend PWA trong `web/`
- Trình hỗ trợ tải dữ liệu trong `scripts/`
- Không gian làm việc dữ liệu cục bộ trong `datasets/` (được git-ignore)

Kho này cũng chứa các không gian làm việc nghiên cứu và tiện ích liên quan (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, submodule `papers`). Runtime chính được mô tả trong README này là ứng dụng `OrganoidAgent` ở thư mục gốc.

## Tính năng ✨

- Lập chỉ mục bộ dữ liệu cục bộ kèm tóm tắt dung lượng và số lượng tệp
- Liệt kê tệp đệ quy trong bộ dữ liệu với suy luận loại tệp
- Hỗ trợ xem trước gồm bảng CSV/TSV/XLS/XLSX
- Hỗ trợ xem trước gồm ảnh TIFF/JPG/PNG
- Hỗ trợ xem trước gồm tóm tắt `.h5ad` với tạo hình xem trước scatter embedding/PCA
- Hỗ trợ xem trước gồm liệt kê tệp nén ZIP/TAR/TGZ + thử xem trước ảnh đầu tiên
- Hỗ trợ xem trước gồm vài dòng đầu của tệp văn bản `.gz`
- Endpoint giải nén archive cho các bộ dữ liệu đóng gói lớn
- Thẻ metadata cấp bộ dữ liệu được render từ Markdown
- Frontend PWA với service worker và manifest
- Làm sạch đường dẫn cơ bản (`safe_dataset_path`) để giới hạn truy cập tệp trong `datasets/`

### Tóm tắt nhanh

| Khu vực | Cung cấp gì |
|---|---|
| Khám phá bộ dữ liệu | Liệt kê bộ dữ liệu theo thư mục với số lượng tệp và tóm tắt dung lượng |
| Khám phá tệp | Liệt kê đệ quy và suy luận loại (`image`, `table`, `analysis`, `archive`, v.v.) |
| Xem trước phong phú | Bảng dữ liệu, TIFF/ảnh, đoạn văn bản gzip, nội dung archive, tóm tắt AnnData |
| Trực quan phân tích | Hình xem trước scatter `.h5ad` từ embedding `obsm` hoặc fallback PCA |
| Hỗ trợ đóng gói | Liệt kê archive + endpoint giải nén cho các gói nén lớn |
| Trải nghiệm web | PWA có thể cài đặt với tài nguyên service worker thân thiện offline |

## Cấu trúc dự án 🗂️

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
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Điều kiện tiên quyết ✅

- Python `3.10+`
- Trình quản lý môi trường được khuyến nghị: `conda` hoặc `venv`

Các gói Python bắt buộc/tùy chọn được suy ra từ mã nguồn:

| Package | Vai trò |
|---|---|
| `tornado` | Bắt buộc để khởi động server |
| `pandas` | Tùy chọn: hỗ trợ xem trước bảng |
| `anndata`, `numpy` | Tùy chọn: xem trước `.h5ad` và vẽ phân tích |
| `Pillow` | Tùy chọn: render ảnh và các bản xem trước được tạo |
| `tifffile` | Tùy chọn: hỗ trợ xem trước TIFF |
| `requests` | Tùy chọn: script tải bộ dữ liệu |
| `kaggle` | Tùy chọn: tải Kaggle trong script drug-screening |

Lưu ý giả định: hiện chưa có `requirements.txt`, `pyproject.toml` hoặc `environment.yml` ở thư mục gốc cho ứng dụng cấp cao nhất.

## Cài đặt ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## Sử dụng 🚀

### Bắt đầu nhanh

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Mở `http://localhost:8080`.

### Kiểm tra API nhanh

```bash
curl http://localhost:8080/api/datasets
```

### Tải dữ liệu (Tùy chọn)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Dữ liệu đã tải sẽ nằm trong `datasets/` (git-ignored).

## API Endpoints 🌐

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/api/datasets` | Liệt kê bộ dữ liệu với thống kê tóm tắt |
| `GET` | `/api/datasets/{name}` | Liệt kê tệp cho một bộ dữ liệu |
| `GET` | `/api/datasets/{name}/metadata` | Trả về thẻ metadata markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Liệt kê tệp theo danh mục |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Payload xem trước theo loại tệp |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Giải nén archive vào thư mục `_extracted` cùng cấp |
| `GET` | `/files/<path>` | Phục vụ tệp dữ liệu thô |
| `GET` | `/previews/<path>` | Phục vụ tài nguyên xem trước đã tạo |

Ví dụ gọi preview:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Cấu hình 🧩

Cấu hình runtime hiện tại được giữ tối giản có chủ đích:

- Cổng server: tham số `--port` trong `app.py` (mặc định `8080`)
- Thư mục dữ liệu: cố định là `datasets/` tương đối với thư mục gốc repo
- Bộ nhớ đệm preview: `datasets/.cache/previews`
- Ánh xạ metadata: dictionary `DATASET_METADATA` trong `app.py`
- GitHub API token cho downloader (tùy chọn): biến môi trường `GITHUB_TOKEN` hoặc `--github-token`

Lưu ý giả định: nếu bạn cần cấu hình dataset root hoặc cài đặt server production, các mục này hiện chưa được công khai trong các tệp cấu hình cấp gốc.

## Ví dụ 🧪

### Duyệt tệp theo danh mục cụ thể

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Giải nén một archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Chạy các chế độ tải chọn lọc

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Ghi chú phát triển 🛠️

- Backend phục vụ tài nguyên tĩnh frontend từ `web/`.
- Service worker và manifest nằm ở `web/sw.js` và `web/manifest.json`.
- Định tuyến theo loại tệp và chức năng xem trước được triển khai trong `app.py`.
- Xác thực thủ công (hướng dẫn hiện tại của dự án): PWA tải tại `http://localhost:8080`
- Xác thực thủ công (hướng dẫn hiện tại của dự án): `/api/datasets` trả về JSON
- Xác thực thủ công (hướng dẫn hiện tại của dự án): bản xem trước render cho CSV/XLSX/images/archives

## Khắc phục sự cố 🩺

- `ModuleNotFoundError` cho thư viện preview: cài các gói còn thiếu (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Danh sách bộ dữ liệu trống: xác nhận dữ liệu tồn tại dưới `datasets/` và thư mục không có tiền tố dấu chấm.
- Preview `.h5ad` thiếu ảnh scatter: kiểm tra đã cài `anndata`, `numpy` và `Pillow`.
- Sự cố preview/giải nén archive lớn: dùng endpoint giải nén và kiểm tra trực tiếp tệp đã giải nén.
- Lỗi giới hạn tốc độ GitHub downloader: cung cấp `GITHUB_TOKEN` qua biến môi trường hoặc cờ CLI.
- Tải Kaggle không hoạt động: cài `kaggle` và cấu hình thông tin xác thực `~/.kaggle/kaggle.json`.

## Lộ trình 🧭

Các cải tiến tiềm năng tiếp theo (chưa được triển khai đầy đủ trong ứng dụng gốc này):

- Thêm manifest phụ thuộc cấp gốc (`requirements.txt` hoặc `pyproject.toml`)
- Thêm kiểm thử tự động cho API handlers và hàm preview
- Thêm cấu hình dataset root và cache
- Thêm hồ sơ chạy production rõ ràng (non-debug, hướng dẫn reverse-proxy)
- Mở rộng tài liệu đa ngôn ngữ trong `i18n/`

## Đóng góp 🤝

Rất hoan nghênh đóng góp. Quy trình thực tế:

1. Fork và tạo một nhánh tập trung.
2. Giữ phạm vi thay đổi trong một khu vực logic.
3. Xác thực thủ công việc khởi động ứng dụng và các endpoint chính.
4. Mở PR với tóm tắt, các lệnh đã chạy và ảnh chụp màn hình cho thay đổi UI.

Quy ước style cục bộ trong repository này:

- Python: thụt lề 4 dấu cách, hàm/tệp dạng snake_case, lớp dạng CapWords
- Giữ logic frontend trong `web/app.js` cho ứng dụng này (tránh viết lại framework không cần thiết)
- Giữ comment ngắn gọn và chỉ thêm khi logic không hiển nhiên

## Bố cục dự án (Tóm tắt chuẩn) 📌

- `app.py`: server Tornado và các route API.
- `web/`: tài nguyên PWA.
- `scripts/`: trình hỗ trợ tải bộ dữ liệu.
- `datasets/`: lưu trữ dữ liệu cục bộ.
- `papers/`: submodule chứa tài liệu tham khảo.

## Giấy phép 📄

Hiện chưa có tệp `LICENSE` cấp dự án ở thư mục gốc của repository này.

Lưu ý giả định: cho đến khi có giấy phép cấp gốc, hãy coi điều khoản tái sử dụng/phân phối lại là chưa được chỉ định cho codebase OrganoidAgent cấp cao nhất.

## Nhà tài trợ & Quyên góp ❤️

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Donate: https://chat.lazying.art/donate
- PayPal: https://paypal.me/RongzhouChen
- Stripe: https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400
