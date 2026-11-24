# Phân loại nhịp ECG bằng TinyML trên STM32

Đồ án vi xử lý VXL-20261 xây dựng hệ thống phát lại ECG từ PC qua UART. STM32 NUCLEO-F446RE dự kiến phát hiện đỉnh R, cắt từng nhịp và chạy 1D-CNN INT8 để dự đoán bốn lớp AAMI (N, S, V, F). PC nhận kết quả và đối chiếu với annotation. Đây là mô hình nghiên cứu dùng dữ liệu ghi sẵn. [Kế hoạch gốc 8 tuần](docs/references/Ke_hoach_do_an_ECG_TinyML_8_tuan.docx)

**Trạng thái khởi tạo.** Có thể tải, kiểm tra và đọc dữ liệu bằng các lệnh dưới đây. Pipeline DSP, mô hình huấn luyện và project CubeIDE là sản phẩm của các tuần tiếp theo.

## Bắt đầu với dữ liệu

Mở terminal tại thư mục chứa README này. Hai bộ dữ liệu dùng cùng định dạng WFDB và được lưu bên ngoài lịch sử Git.

| Thư mục | Nội dung và mục đích |
| --- | --- |
| `data/raw/mitdb/` | 48 bản ghi ECG hai kênh lấy mẫu ở 360 Hz, dùng cho huấn luyện và đánh giá phân loại nhịp. [MIT-BIH Arrhythmia Database 1.0.0](https://physionet.org/content/mitdb/1.0.0/) |
| `data/raw/nstdb/` | 12 bản ghi ECG có nhiễu tạo từ 118 và 119, cùng ba bản ghi nhiễu riêng (bw, ma, em). Dùng cho thử nghiệm độ bền. [MIT-BIH Noise Stress Test Database 1.0.0](https://physionet.org/content/nstdb/1.0.0/) |

**Kiểm tra bản trên máy** bằng Python 3, không cần cài thư viện hay kết nối mạng. Mỗi bộ phải in một dòng `SHA-256 OK`.

```bash
python3 src/dataset/download.py --verify
```

**Trên bản clone mới hoặc khi thiếu/hỏng tệp**, chạy lệnh bên dưới. Script tải hai gói ZIP phiên bản 1.0.0 từ PhysioNet, kiểm tra checksum trước khi chép vào `data/raw/` và kiểm tra lại tại đích. Lệnh này cũng thay thế tệp cùng tên nếu đã có. Khi trích dẫn dữ liệu, dùng thông tin tác giả, giấy phép và DOI trên hai trang nguồn.

```bash
python3 src/dataset/download.py
```

**Đọc thử 10 giây của bản ghi 100.** Tệp `.hea` chứa thông số thu và tên đạo trình, `.dat` chứa mẫu, `.atr` chứa annotation tham chiếu. Truyền cho WFDB đường dẫn cơ sở, bỏ phần mở rộng. Ví dụ sau lưu hình ECG có dấu tại vị trí annotation vào `results/figures/100.png`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
import wfdb
import matplotlib.pyplot as plt

base = "data/raw/mitdb/100"
record = wfdb.rdrecord(base, sampto=3600, channel_names=["MLII"])
ann = wfdb.rdann(base, "atr", sampto=3599)
signal = record.p_signal[:, 0]
plt.plot([i / record.fs for i in range(len(signal))], signal)
plt.scatter(ann.sample / record.fs, signal[ann.sample], s=15)
plt.xlabel("Thời gian (s)")
plt.ylabel(f"MLII ({record.units[0]})")
plt.tight_layout()
plt.savefig("results/figures/100.png")
print(record.fs, record.sig_name, ann.symbol)
PY
```

## Tổ chức thư mục

| Đường dẫn | Lưu gì ở đây |
| --- | --- |
| `data/raw/`, `data/processed/`, `data/splits/` | Bản gốc, dữ liệu đã xử lý, danh sách bản ghi chia tập. Commit danh sách chia tập để tái lập thí nghiệm. |
| `notebooks/`, `configs/` | Notebook khảo sát và cấu hình thí nghiệm (dataset, DSP, model, training, quantization, hardware). |
| `src/dataset/`, `src/dsp/`, `src/features/` | Tải/đọc dữ liệu, ánh xạ nhãn, lọc và phát hiện R, cắt nhịp và tính đặc trưng RR. |
| `src/models/`, `src/training/`, `src/optimization/`, `src/evaluation/` | Kiến trúc, huấn luyện, lượng tử hóa và đánh giá mô hình trên PC. |
| `firmware/stm32f446re/` | Core và Drivers cho mã sinh bởi CubeMX/HAL, Middlewares cho thư viện. App chia thành Communication, Buffer, DSP, Features, SQI, AI, Logging và Profiling. |
| `models/{fp32,int8,exported}/` | Trọng số FP32, mô hình INT8 và mảng C xuất từ cùng tệp TFLite. Ghi SHA-256 khi bàn giao. |
| `tools/`, `tests/golden_vectors/` | Công cụ PC phát UART, nhận log, xuất mô hình và dữ liệu kiểm thử đối chiếu Python/MCU. |
| `results/` | Báo cáo baseline, fp32, int8, hil, robustness, profiling và hình vẽ trong figures. |
| `docs/`, `.github/` | Tài liệu tham khảo, kiến trúc, báo cáo tuần, báo cáo cuối và cấu hình cộng tác GitHub. |

## Quy trình xử lý và đánh giá

1. Chọn đạo trình MLII theo tên trong header. Bỏ các bản ghi 102, 104, 107, 217 khi tạo tập phân loại. Chia train/validation theo bản ghi trong DS1. Chốt `data/splits/split_v1.json` trước khi huấn luyện. Giữ DS2 đến tuần 8 và chỉ dùng DS1 để hiệu chuẩn INT8.
2. Trên PC, chuyển 360 Hz về 250 Hz và đóng gói int16. Khi so khớp với annotation, quy đổi chỉ số mẫu về cùng tần số. Trước khi kết nối board, thống nhất hệ số int16/mV, byte order và định dạng khung. Baud rate đề xuất là 921600.
3. Trên MCU, nhận bằng UART DMA qua ring buffer 1024 mẫu, lọc IIR nhân quả 0,5–40 Hz và phát hiện QRS bằng Pan–Tompkins. Bản tham chiếu Python dùng SOS nhân quả. Reset trạng thái giữa các bản ghi và đếm số lần tràn bộ đệm.
4. Mỗi đầu vào CNN gồm 90 mẫu trước R, 166 mẫu tính từ R và bốn đặc trưng RR (trước, sau, trung bình 10 nhịp, dài hạn 5 phút). Chuẩn hóa RR theo giá trị dài hạn. Ghi kết quả N/S/V/F hoặc trạng thái REJECT_SQI, UNCERTAIN.
5. Trong HIL (hardware-in-the-loop), PC lưu CSV có record, r_index, pred, xác suất, SQI, status, latency và model_version. Đối chiếu annotation trên PC. Báo cáo riêng thời gian đợi cửa sổ, đợi RR sau và thời gian xử lý khi đủ đầu vào.

## Mốc làm việc và nghiệm thu

| Tuần | Đầu ra theo kế hoạch |
| --- | --- |
| 1 | Khảo sát ECG/annotation, thống kê nhãn, khóa split, thử Blink LED. [Báo cáo tuần 1](docs/weekly_reports/week_01.md) |
| 2 | Resample, đặc tuyến bộ lọc, kiểm tra truyền/nhận đúng mẫu int16. |
| 3 | Đối chiếu đỉnh R, tạo DS1.npz, kiểm thử DMA liên tục. Mục tiêu Se và PPV detector ≥99% trên DS1. |
| 4 | Baseline và CNN Small FP32, validation theo nhóm bản ghi, so bộ lọc Python/C. |
| 5 | Chốt FP32, SQI và ngưỡng từ chối trên DS1, đối chiếu DSP trên MCU. |
| 6 | Lượng tử hóa và thử 200 golden vectors. Mục tiêu sai lệch đầu ra ≤1 LSB. |
| 7 | Tích hợp HIL trên DS1, đo tài nguyên và khóa cấu hình. Mục tiêu ≥99% nhãn khớp giữa Python/MCU. |
| 8 | Đánh giá FP32/INT8 PC/INT8 MCU trên DS2, kiểm thử NSTDB ở 24/18/12/6/0 dB, bàn giao báo cáo, slide và demo. |

**Mục tiêu cuối kỳ.** Báo cáo cả DS2 gốc và DS2 bỏ 202 (kết quả chính vì 201/202 cùng người). Mục tiêu Macro-F1 ≥0,75, giảm ≤0,02 khi chuyển FP32 sang INT8, Flash mô hình + kernel ≤250 KB, RAM inference ≤100 KB và độ trễ xử lý ≤200 ms. Kèm sensitivity, PPV, F1 từng lớp, balanced accuracy, confusion matrix, ECE, coverage, tải CPU và latency mean/p95/max.

**Làm việc nhóm.** Theo kế hoạch, phát triển trên nhánh `feature/<nhiệm-vụ>`, tích hợp qua PR vào `develop`, giữ bản ổn định trên `main`. Đưa cấu hình, split đã chốt, mã nguồn, file .ioc, model card và kết quả tổng hợp vào Git. Dataset, checkpoint, build tạm và log lớn lưu cục bộ.
