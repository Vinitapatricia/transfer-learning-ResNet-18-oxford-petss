# Transfer Learning ResNet-18 pada Oxford-IIIT Pet (10 Kelas)

Tugas GSLC Deep Learning – [Nama], [NIM], [Kelas], Binus University.

## Deskripsi
Perbandingan empat kondisi pelatihan ResNet-18 pada subset 10 kelas Oxford-IIIT Pet:
- **B1** Baseline from scratch
- **B2** Feature extraction (backbone beku, hanya `fc`)
- **B3** Fine-tuning `layer4` + `fc` dengan differential learning rate
- **B4** Kontrol augmentasi (tanpa vs dengan augmentasi, pada konfigurasi B3)
- **B5** EfficientNet-B0, C5 Grad-CAM

## Dataset
Oxford-IIIT Pet, 10 kelas: Abyssinian, Bengal, Birman, Bombay, Persian, Beagle, Boxer, Chihuahua, Pug, Samoyed.
Total 1981 gambar, split stratified 70/15/15 (1386 / 297 / 298) dengan seed 42.
Dataset diunduh otomatis oleh torchvision.

## Hasil Ringkas
| |Strategi| Trainable Params	| Learning Rate |	Epoch Konvergen	| Durasi (s) | Val Acc | Gap Test | Macro F1|
|B1|Semua layer dilatih dan model diberi boobt acak |	11181642 | [0.001]| 6|	3818.084|	0.4916|	0.0394|	0.4876|
|B2|	Seluruh layer backbone dibekukan dan layer fc dilatih|	5130|	[0.001]|	24|	4999.999|	0.9798|	0.0166|	0.9634|
|B3|	layer 4 dan fc dilatih dengan differential learning rate	|8398858|	[0.0001, 0.001]|	4	|1663.058|	0.9798	|0.0202	|0.9800|
|B4|	Melatih model B3 tanpa augmentasi (hanya Resize/Crop dan Normalisasi)|8398858|	[0.0001, 0.001]	|3	|1428.836|	0.9764|	0.0214|	0.9765|
|B4|	Melatih model B3 dengan augmentasi (Resize/Crop, Random Horizontal Flip, Random Rotation 15°, dan Color Jitter)|	8398858|	[0.0001, 0.001]|	12|	3722.132|	0.9697|	0.0289|	0.9602|
|B5|	Melatih model dari B3 pada arsitektur EfficientNet-B0|	1142202	|[0.0001, 0.001]|	10|	2162.399|	0.9933|	0.0031|	0.9796|


## Hyperparameter
Batch size 32, AdamW, CrossEntropyLoss, early stopping (patience 3, pada val loss),
seed 42. B3: LR 1e-4 (layer4) dan 1e-3 (fc).

## Cara Menjalankan
1. Clone repo: `git clone <url-repo>`
2. `pip install -r requirements.txt`
3. Buka `GSLC_DeepLearning.ipynb` (Colab disarankan, GPU) lalu **Run All**.
4. Log tersimpan otomatis ke `log_per_epoch.csv`, `log_ringkasan.csv`, `log_hasil.json`

## Isi Repo
- `GSLC_DeepLearning.ipynb` – seluruh eksperimen
- `laporan.pdf` – laporan analisis
- `Bagian A.pdf` - jawaban untuk bagian A
- `log_per_epoch.csv` - log mentah per epoch
- `log_ringkasan.csv` - log ringkasan
- `log_hasil.json` - log hasil

