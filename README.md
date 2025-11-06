# Snake ID – Myanmar (CPU-only MVP)

This is a minimal, **CPU-only** snake identifier for Myanmar. It classifies a photo into a small set of species and returns Burmese safety guidance with conservation-first language.

## Quick Start

1. Create a virtual env and install requirements:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Organize your dataset (CC-licensed images) as:
   ```text
   data/images/
     Naja_kaouthia/
       img001.jpg
       ...
     Ophiophagus_hannah/
     Bungarus_fasciatus/
     Daboia_russelii/
     Trimeresurus_spp/
     Ahaetulla_fronticincta/
     Ptyas_mucosa/
     Python_bivittatus/
   ```

3. (Optional) Use the downloader to fetch CC images from iNaturalist by species list:
   ```bash
   python tools/download_inat.py --species_csv data/species_list.csv --out_dir data/raw_inat --license CC-BY,CC0
   python tools/prepare_dataset.py --raw_dir data/raw_inat --out_dir data/images
   ```

4. Train and export a compact CPU model:
   ```bash
   python train_cpu.py --data_dir data/images --out_dir artifacts --arch mobilenetv3_small --epochs 8
   ```

   This will:
   - train with transfer learning
   - export ONNX (`artifacts/model_fp32.onnx`)
   - quantize to INT8 (`artifacts/model_int8.onnx`)
   - save label map (`artifacts/class_map.json`)

5. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Open http://127.0.0.1:8000 in your browser (Burmese UI).

## Burmese Safety Messaging

Safety cards are in `app/safety_cards/mm.json` and can be edited by community experts. All bite events: **seek medical care immediately**.

## Notes

- This MVP targets 8 classes to keep it small. You can add more classes once you have enough data.
- Use only CC-licensed images and keep attribution (iNaturalist/Wikimedia). See `data/ATTRIBUTION.csv`.
