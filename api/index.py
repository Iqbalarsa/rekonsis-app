from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from pathlib import Path
import io
import json  # <--- Ini yang tadi ketinggalan
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

app = Flask(__name__)

# --- LOAD AI ASSETS ---
model = tf.keras.models.load_model(str(ROOT_DIR / 'model_susenas.h5'), compile=False)

with open(ROOT_DIR / 'scaler_susenas.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(ROOT_DIR / 'all_coicop.pkl', 'rb') as f:
    all_coicop = [str(c).strip() for c in pickle.load(f)] # Pastikan master COICOP adalah String

# --- LOAD MASTER KOMODITAS ---
# Tambahkan converters agar COICOP dibaca sebagai String, bukan angka
df_master = pd.read_excel(ROOT_DIR / 'master_komoditas.xlsx', dtype={'COICOP': str})
df_master['COICOP'] = df_master['COICOP'].str.strip()
mapping_komoditas = dict(zip(df_master['COICOP'], df_master['Nama_Komoditas']))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # 1. BACA EXCEL DENGAN PAKSAAN STRING
            # Ini kunci agar nol di depan tidak hilang
            df_test_long = pd.read_excel(file, dtype={'COICOP': str})
            
            # Bersihkan data
            df_test_long['COICOP'] = df_test_long['COICOP'].str.strip()
            df_test_long['B41K9'] = pd.to_numeric(df_test_long['B41K9'], errors='coerce').fillna(0)
            
            # 2. PROSES AI
            df_test_wide_raw = df_test_long.groupby('COICOP')['B41K9'].sum()
            
            # Penyelarasan (Reindex)
            df_test_wide = df_test_wide_raw.reindex(all_coicop, fill_value=0)
            X_test_raw = df_test_wide.values.reshape(1, -1)
            
            # Prediksi
            X_test_scaled = scaler.transform(np.log1p(X_test_raw))
            reconstructed_scaled = model.predict(X_test_scaled)
            recon_asli = np.expm1(scaler.inverse_transform(reconstructed_scaled))
            diff_scaled = np.abs(X_test_scaled[0] - reconstructed_scaled[0])
            
            # 3. DataFrame Hasil
            df_res = pd.DataFrame({
                'Kode_COICOP': all_coicop,
                'Input': X_test_raw[0],
                'Rekomendasi': recon_asli[0],
                'LogDiff': diff_scaled.round(4)
            })

            # 4. Mapping Nama
            df_res['Komoditas'] = df_res['Kode_COICOP'].map(mapping_komoditas).fillna("Tidak Diketahui")

            # 5. Status
            def get_status(row):
                if row['Input'] == 0 and row['LogDiff'] > 0.15:
                    return "ANOMALI (Lupa Isi)", "table-danger"
                elif row['LogDiff'] > 0.15:
                    return "ANOMALI (Jarak)", "table-warning"
                return "WAJAR", ""

            df_res[['STATUS', 'CLASS']] = df_res.apply(lambda r: pd.Series(get_status(r)), axis=1)
            
            # 6. FILTER & FORMATTING
            df_filtered = df_res[(df_res['Input'] > 0) | (df_res['STATUS'] != "WAJAR")].copy()
            
            # Formatting untuk tampilan (2 digit)
            df_filtered['Input_Disp'] = df_filtered['Input'].map('{:,.2f}'.format)
            df_filtered['Rekomendasi_Disp'] = df_filtered['Rekomendasi'].map('{:,.2f}'.format)
            
            hasil = df_filtered.to_dict('records')
            return render_template('index.html', results=hasil)
            
    return render_template('index.html', results=None)

@app.route('/download-template')
def download_template():
    return send_from_directory(directory=ROOT_DIR, 
                               path='template.xlsx', 
                               as_attachment=True)
    
@app.route('/download-hasil', methods=['POST'])
def download_hasil():
    data_json = request.form.get('results_data')
    if not data_json:
        return "Data tidak ditemukan", 400
    
    data_list = json.loads(data_json)
    df_hasil = pd.DataFrame(data_list)
    
    # Pilih kolom untuk di-export
    kolom_final = ['Kode_COICOP', 'Komoditas', 'Input', 'Rekomendasi', 'STATUS']
    df_export = df_hasil[kolom_final]
    
    # --- PROSES TIMESTAMP ---
    # format: TahunBulanTanggal_JamMenitDetik (Contoh: 20240520_143005)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nama_file = f"Hasil_Validasi_{timestamp}.xlsx"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Hasil Validasi')
    
    output.seek(0)
    
    # Kirim file dengan nama yang ada timestamp-nya
    from flask import send_file # Pastikan sudah di-import
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=nama_file
    )
    
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True)