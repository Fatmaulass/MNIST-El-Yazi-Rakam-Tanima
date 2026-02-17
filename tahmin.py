import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import joblib
import os

# 1. MODEL MİMARİSİ (MLP İÇİN)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. AYARLAR VE DOSYA YOLLARI
MLP_MODEL_PATH = "mnist_mlp_model.pth"
RF_MODEL_PATH = "mnist_rf_model.pkl"
RF_SCALER_PATH = "mnist_scaler.pkl"
device = torch.device("cpu")

# MLP için dönüşüm işlemleri
transform_mlp = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3. BİRLEŞTİRİLMİŞ ARAYÜZ SINIFI
class UnifiedDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Karşılaştırma: MLP vs Random Forest")
        
        # Değişkenler
        self.mlp_model = None
        self.rf_model = None
        self.rf_scaler = None
        self.is_mlp_ready = False
        self.is_rf_ready = False

        # Modelleri Yükle
        self.load_models()

        # --- ARAYÜZ TASARIMI ---
        
        # 1. Çizim Alanı
        self.canvas_width = 300
        self.canvas_height = 300
        
        # Siyah zemin (MNIST formatı için)
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor="cross")
        self.canvas.pack(pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self.paint) # Çizim bağlama

        # Bellekteki resim (Modelin okuyacağı)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 2. Butonlar
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        self.btn_predict = tk.Button(btn_frame, text="İKİ MODELLE TAHMİN ET", command=self.predict_both, 
                                     bg="#007bff", fg="white", font=("Arial", 11, "bold"), width=25)
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        
        self.btn_clear = tk.Button(btn_frame, text="TEMİZLE", command=self.clear_canvas, 
                                   bg="#dc3545", fg="white", font=("Arial", 11, "bold"), width=10)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 3. Sonuç Alanı (Frame içinde iki ayrı etiket)
        result_frame = tk.Frame(root, relief=tk.RIDGE, borderwidth=2)
        result_frame.pack(pady=15, padx=10, fill=tk.X)

        # Başlıklar
        tk.Label(result_frame, text="MLP (Derin Öğrenme)", font=("Arial", 10, "bold", "underline"), fg="blue").grid(row=0, column=0, padx=20, pady=5)
        tk.Label(result_frame, text="Random Forest", font=("Arial", 10, "bold", "underline"), fg="blue").grid(row=0, column=1, padx=20, pady=5)

        # Değerler
        self.lbl_mlp_result = tk.Label(result_frame, text="Sonuç: -", font=("Helvetica", 14))
        self.lbl_mlp_result.grid(row=1, column=0, padx=20, pady=10)

        self.lbl_rf_result = tk.Label(result_frame, text="Sonuç: -", font=("Helvetica", 14))
        self.lbl_rf_result.grid(row=1, column=1, padx=20, pady=10)
        
        # Grid sütunlarını ortala
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_columnconfigure(1, weight=1)

    def load_models(self):
        """Tüm modelleri yüklemeyi dener"""
        print("--- Modeller Yükleniyor ---")
        
        # 1. MLP Yükleme
        try:
            if os.path.exists(MLP_MODEL_PATH):
                self.mlp_model = SimpleMLP().to(device)
                self.mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device))
                self.mlp_model.eval()
                self.is_mlp_ready = True
                print(f"[OK] MLP Modeli yüklendi.")
            else:
                print(f"[HATA] {MLP_MODEL_PATH} bulunamadı.")
        except Exception as e:
            print(f"[HATA] MLP yüklenirken sorun: {e}")

        # 2. Random Forest Yükleme
        try:
            if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                self.rf_scaler = joblib.load(RF_SCALER_PATH)
                self.is_rf_ready = True
                print(f"[OK] Random Forest ve Scaler yüklendi.")
            else:
                print(f"[HATA] RF model veya scaler dosyası eksik.")
        except Exception as e:
            print(f"[HATA] RF yüklenirken sorun: {e}")

    def paint(self, event):
        """Çizim fonksiyonu"""
        r = 10 # Fırça boyutu
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        
        # Ekrana çiz
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        # Belleğe çiz
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_mlp_result.config(text="Sonuç: -", fg="black")
        self.lbl_rf_result.config(text="Sonuç: -", fg="black")

    def predict_both(self):
        """Resmi alır ve her iki modele gönderir"""
        
        # Ortak Ön İşleme: Resmi küçült
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # TAHMİN 1: MLP (PyTorch)
        if self.is_mlp_ready:
            try:
                # Tensor dönüşümü ve normalizasyon
                input_tensor = transform_mlp(img_resized).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = self.mlp_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    pred_mlp = torch.argmax(probs, dim=1).item()
                    conf_mlp = probs[0][pred_mlp].item() * 100
                
                self.lbl_mlp_result.config(text=f"{pred_mlp}\n(Güven: %{conf_mlp:.1f})", fg="green")
            except Exception as e:
                self.lbl_mlp_result.config(text="Hata", fg="red")
                print(f"MLP Hatası: {e}")
        else:
            self.lbl_mlp_result.config(text="Model Yok", fg="gray")

        # TAHMİN 2: Random Forest (Sklearn)
        if self.is_rf_ready:
            try:
                # Numpy dönüşümü, düzleştirme ve scale
                img_array = np.array(img_resized)
                img_flattened = img_array.reshape(1, -1)
                img_normalized = self.rf_scaler.transform(img_flattened.astype(np.float32))
                
                pred_rf = self.rf_model.predict(img_normalized)[0]
                probs_rf = self.rf_model.predict_proba(img_normalized)[0]
                conf_rf = probs_rf[pred_rf] * 100
                
                self.lbl_rf_result.config(text=f"{pred_rf}\n(Güven: %{conf_rf:.1f})", fg="green")
            except Exception as e:
                self.lbl_rf_result.config(text="Hata", fg="red")
                print(f"RF Hatası: {e}")
        else:
            self.lbl_rf_result.config(text="Model Yok", fg="gray")

# Uygulamayı Başlat
if __name__ == "__main__":
    root = tk.Tk()
    
    # Pencereyi ortala
    w, h = 500, 600
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = int((ws/2) - (w/2))
    y = int((hs/2) - (h/2))
    root.geometry(f"{w}x{h}+{x}+{y}")
    
    app = UnifiedDigitRecognizer(root)
    root.mainloop()