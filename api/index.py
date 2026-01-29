from flask import Flask, render_template, request, send_from_directory, send_file
import pandas as pd
import numpy as np
import pickle
import os
import io
import json
from datetime import datetime
from pathlib import Path

# --- LOAD TFLITE RUNTIME ---
# Kita gunakan try-except agar bisa jalan di laptop (TF full) maupun di Vercel (TFLite)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

app = Flask(__name__)

# --- LOAD AI ASSETS (VERSI LITE) ---
# Menggunakan Interpreter alih-alih tf.keras.models.load_model
interpreter = tflite.Interpreter(model_path=str(ROOT_DIR / 'model_susenas.tflite'))
interpreter.allocate_tensors()

# Ambil informasi input dan output untuk proses prediksi nanti
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(ROOT_DIR / 'scaler_susenas.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(ROOT_DIR / 'all_coicop.pkl', 'rb') as f:
    all_coicop = [str(c).strip() for c in pickle.load(f)]

# --- LOAD MASTER KOMODITAS ---
df_master = pd.read_excel(ROOT_DIR / 'master_komoditas.xlsx', dtype={'COICOP': str})
df_master['COICOP'] = df_master['COICOP'].str.strip()
mapping_komoditas = dict(zip(df_master['COICOP'], df_master['Nama_Komoditas']))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # 1. BACA EXCEL
            df_test_long = pd.read_excel(file, dtype={'COICOP': str})
            df_test_long['COICOP'] = df_test_long['COICOP'].str.strip()
            df_test_long['B41K9'] = pd.to_numeric(df_test_long['B41K9'], errors='coerce').fillna(0)
            
            # 2. PROSES DATA
            df_test_wide_raw = df_test_long.groupby('COICOP')['B41K9'].sum()
            df_test_wide = df_test_wide_raw.reindex(all_coicop, fill_value=0)
            X_test_raw = df_test_wide.values.reshape(1, -1)
            
            # 3. PREDIKSI (LOGIKA TFLITE)
            # Scaling & pastikan tipe data float32 agar TFLite tidak error
            X_test_scaled = scaler.transform(np.log1p(X_test_raw)).astype(np.float32)
            
            # Jalankan Model
            interpreter.set_tensor(input_details[0]['index'], X_test_scaled)
            interpreter.invoke()
            
            # Ambil Hasil Prediksi
            reconstructed_scaled = interpreter.get_tensor(output_details[0]['index'])
            
            # Balikkan ke Nilai Asli
            recon_asli = np.expm1(scaler.inverse_transform(reconstructed_scaled))
            diff_scaled = np.abs(X_test_scaled[0] - reconstructed_scaled[0])
            
            # 4. DataFrame Hasil
            df_res = pd.DataFrame({
                'Kode_COICOP': all_coicop,
                'Input': X_test_raw[0],
                'Rekomendasi': recon_asli[0],
                'LogDiff': diff_scaled.round(4)
            })

            df_res['Komoditas'] = df_res['Kode_COICOP'].map(mapping_komoditas).fillna("Tidak Diketahui")

            def get_status(row):
                if row['Input'] == 0 and row['LogDiff'] > 0.15:
                    return "ANOMALI (Lupa Isi)", "table-danger"
                elif row['LogDiff'] > 0.15:
                    return "ANOMALI (Jarak)", "table-warning"
                return "WAJAR", ""

            df_res[['STATUS', 'CLASS']] = df_res.apply(lambda r: pd.Series(get_status(r)), axis=1)
            
            df_filtered = df_res[(df_res['Input'] > 0) | (df_res['STATUS'] != "WAJAR")].copy()
            df_filtered['Input_Disp'] = df_filtered['Input'].map('{:,.2f}'.format)
            df_filtered['Rekomendasi_Disp'] = df_filtered['Rekomendasi'].map('{:,.2f}'.format)
            
            hasil = df_filtered.to_dict('records')
            return render_template('index.html', results=hasil)
            
    return render_template('index.html', results=None)

@app.route('/download-template')
def download_template():
    return send_from_directory(directory=ROOT_DIR, path='template.xlsx', as_attachment=True)
    
@app.route('/download-hasil', methods=['POST'])
def download_hasil():
    data_json = request.form.get('results_data')
    if not data_json: return "Data tidak ditemukan", 400
    
    data_list = json.loads(data_json)
    df_hasil = pd.DataFrame(data_list)
    df_export = df_hasil[['Kode_COICOP', 'Komoditas', 'Input', 'Rekomendasi', 'STATUS']]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nama_file = f"Hasil_Validasi_{timestamp}.xlsx"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Hasil Validasi')
    output.seek(0)
    
    return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name=nama_file)
    
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True)