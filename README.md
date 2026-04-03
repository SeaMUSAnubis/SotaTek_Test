# README — BOM Document Intelligence Pipeline

## Tổng quan

Pipeline này tự động xử lý ảnh tài liệu BOM (Bill of Materials) kỹ thuật qua 4 bước:

```
Ảnh đầu vào
    │
    ▼
[1] Detection (Faster R-CNN / Detectron2)
    │   Phân loại: PartDrawing | Table | Note
    │
    ▼
[2] NMS + Overlap Filter (IoA > 70%)
    │   Loại bỏ bounding-box trùng lặp
    │   Giữ box có confidence cao hơn (hoặc diện tích lớn hơn)
    │
    ▼
[3] Crop ảnh theo từng object
    │   Lưu vào thư mục con: output/Note/, output/Table/, output/PartDrawing/
    │
    ▼
[4] OCR (Donut VisionEncoderDecoder)
        → Note  : trích xuất plain-text (Chỉ dùng được cho tiếng Anh)
        → Table : trích xuất và giữ nguyên format bảng (Chưa làm được)
        → PartDrawing : không OCR
    │
    ▼
JSON kết quả + ảnh crop
```

---

## Cấu trúc thư mục

```
Code_all/
├── pipeline.py             ← File script chính để chạy toàn bộ pipeline
├── Detection_image.ipynb   ← Notebook hướng dẫn training Detectron2 cho bài toán nhận diện
├── OCR_note_table.ipynb    ← Notebook hướng dẫn training Donut OCR cho bài toán trích xuất văn bản
├── requirements.txt        ← File liệt kê các thư viện cần thiết
└── README.md               ← Hướng dẫn sử dụng
```

---

## Yêu cầu cài đặt

### 1. Môi trường Python

- `Python >= 3.10`
- `CUDA >= 11.3` (khuyến nghị có GPU để chạy mô hình nhanh hơn, có thể chạy CPU)

### 2. Cài đặt thư viện (requirements)

Cài đặt bằng `requirements.txt` hoặc cài thủ công các package sau:

```bash
# Cài đặt PyTorch (ví dụ cho CUDA 11.8 - chọn đúng phiên bản với máy của bạn)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài đặt Detectron2 (build từ source, yêu cầu phải tương thích với version CUDA)
pip install git+https://github.com/facebookresearch/detectron2.git

# Cài đặt HuggingFace và OCR Donut, các thư viện xử lý ảnh
pip install -r requirements.txt
# Hoặc: pip install transformers datasets accelerate evaluate sentencepiece opencv-python Pillow numpy
```

---

## Hướng dẫn Inference

Chạy file `pipeline.py` bằng Command Line để lấy kết quả (Detection + OCR).

### Câu lệnh cơ bản

```bash
python pipeline.py --image path/to/image.jpg
```
*Lưu ý: Thay đổi dir weight cho model Detectron2 tại `Weight/Detection/detection_weight.pth` và model OCR Donut tại `Weight/OCR`.

image: Đường dẫn tới file ảnh đầu vào (jpg/png) 
Detectron2_weight: Đường dẫn tới file `.pth` của model Detectron2 (Ví dụ:`Weight/Detection/detection_weight.pth`)
OCR_weight: Đường dẫn thư mục chứa model OCR Donut đã huấn luyện  (`Weight/OCR`)
output_dir: Thư mục lưu kết quả (ảnh crop và file JSON)
score_thresh: Ngưỡng confidence tối thiểu cho detection
overlap_thresh: Ngưỡng IoA (Intersection over Area) để lọc bỏ bounding box trùng lặp (NMS) | `0.70` |

## Output / Chẩn đoán Kết quả

Khi hoàn tất, script sẽ tạo trong thư mục `--output-dir` (thường là `output/`):
1. **Thư mục theo Entity:**
   - `output/PartDrawing/`
   - `output/Table/`
   - `output/Note/`
   *(Chứa các hình ảnh đã được crop cho từng đối tượng)*
2. **File JSON kết quả `[tên_ảnh]_results.json` có cấu trúc:**
   ```json
   {
     "image": "path/to/image.jpg",
     "total_objects": 3,
     "objects": [
       {
         "id": 1,
         "class": "Table",
         "confidence": 0.98,
         "bbox": {"x1": 100, "y1": 200, "x2": 600, "y2": 800},
         "crop_path": "./output/Table/image_Table_1.jpg",
         "ocr_content": "<tr><td>Cell 1</td>...</tr>"
       }
     ]
   }
   ```

---

## Tuỳ Chỉnh (Training)

Nếu bạn muốn training lại các mô hình trên dữ liệu riêng của mình, có thể tham khảo trực tiếp mã nguồn trong Jupyter Notebook.

1. `Detection_image.ipynb`: Chứa pipeline tiền xử lý ảnh COCO cho Detectron2, split dữ liệu train/val/test, và config để fine-tune mô hình *Faster R-CNN R50-FPN*.
2. `OCR_note_table.ipynb`: Cấu hình quá trình Fine-Tune model *Naver-Clova Donut Base* để xử lý OCR text và table structure. (Bao gồm Data Collator đặc biệt, gradient accumulation, config token ID,... để tránh OOM trên GPU yếu).

### Lý do sử dụng những công nghệ này
- Detectron2 là một hệ thống mô hình.
Nói cách khác khi chạy một mô hình phát hiện vậy thể (cụ thể trong bài em sử dụng là Faster R-CNN) trong detectron2 thì mô hình sẽ đi qua 3 khối đó là 
    Backbone: Dùng để extract feature maps (R50-FPN)
    Proposal Generator: Ở đây mô hình sẽ tìm ra các vùng có khả năng là vật thể khi đó nó sẽ tạo ra anchor boxes để dự đoán trong cái anchor boxes này chứa cái gì
    ROI heads: Tại khối này mô hình sẽ nhận gợi ý từ khối 2 và đưa ra kết quả cuối cùng
Tuy nhiên dù được trải qua nhiều lớp sàng lọc và đánh giá model vẫn có nhiều bounding box chồng chéo lẫn nhau. Để giảm thiểu điều này em sử dụng thêm thuật toán NMS giúp loại bỏ các bounding box trung lặp.

- Donut là có tốc độ suy luận nhanh, do không cần chạy các module OCR vốn tiêu tốn nhiều tài nguyên, dễ dàng tinh chỉnh (Fine-tuning) Người dùng có thể dễ dàng tinh chỉnh mô hình cho các loại tài liệu cụ thể thông qua các thư viện phổ biến như Hugging Face và đặc biệt khi tự huấn luyện (train) hoặc tinh chỉnh (fine-tune) Donut trên tập dữ liệu riêng thì có toàn quyền sở hữu và khai thác thương mại mô hình đó.

### Hạn chế 
1. Dù rằng đã có những cải tiến giúp loại bỏ các bounding box không cần thiếu nhưng model vẫn không thể phân biệt được table và note một cách cụ thể.
Nhận xét: Do việc phải tự label mà không có chuyên gia giúp khoanh vùng + dữ liệu khá ít về note và table nên model không học được cách phân biệt hai labels đó nhiều.
2. Về việc OCR em đã cố gắng tìm những nguồn dữ liệu có ảnh mờ bằng tiếng Việt để fine-tuning những không tìm được. Do đó em fine-tuning trên dữ liệu tiếng Anh, cho thấy model nhận diện chữ tiếng Anh rất chính xác. Khi chuyển sang nhận diện bằng tiếng Việt lại không ổn.
Nhận xét: Hạn chế về mặt phần cứng, em traning model bằng kaggle nên dữ liệu nhiều cũng khó có thể load và traning được. Nên em chỉ dùng 5% số lượng data và giới hạn token về mức tối thiểu để GPU có thể chạy được mà không bị tràn ram.

### Hướng cải tiến
1. Điều chỉnh lại các kiến trúc của model, hướng này giúp cho ta nhận biết được những lớp nào trong model khiến cho việc phân biệt 3 labels khó khăn
2. Tìm thêm số lượng data và dành nhiều thời gian cho việc labels data (Vì labels có tốt thì model mới thực sự biết được cách phân biệt 3 labels dễ dàng)


---

## Liên hệ & nguồn tham khảo

- **Detectron2**: https://github.com/facebookresearch/detectron2
- **Donut**: https://github.com/clovaai/donut
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
