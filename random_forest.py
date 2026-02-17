import numpy as np
import matplotlib.pyplot as plt
import joblib
from torchvision import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

def main():
    # 1. VERİ YÜKLEME VE ÖN İŞLEME
    train_data = datasets.MNIST(root='./data', train=True, download=True)
    test_data  = datasets.MNIST(root='./data', train=False, download=True)

    # Eğitim verisi (Düzleştirilmiş)
    X_train = train_data.data.numpy().reshape(-1, 28 * 28)
    y_train = train_data.targets.numpy()

    # Test verisini İKİ formatta tutuyoruz:
    # 1. Model tahmini için DÜZLEŞTİRİLMİŞ (Vektör)
    X_test_flat = test_data.data.numpy().reshape(-1, 28 * 28)
    # 2. Görselleştirme ve LIME için HAM RESİM (Matris)
    X_test_img = test_data.data.numpy()
    y_test = test_data.targets.numpy()

    # Normalizasyon (Scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.transform(X_test_flat.astype(np.float32))

    print(f" -> Eğitim Seti Boyutu: {X_train_scaled.shape}")

    # 2. MODEL EĞİTİMİ
    print("Model Eğitiliyor (Random Forest)")

    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    print(" -> Eğitim Tamamlandı.")

    # 3. TEST VE PERFORMANS RAPORU
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"TEST SONUCU (Accuracy): %{acc * 100:.2f}")
    print("-" * 30)
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    # Görsel 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"Karmaşıklık Matrisi (Başarı: %{acc*100:.2f})")
    plt.tight_layout()
    plt.show()

    # Görsel 2: Hatalı Tahmin Analizi
    hatali_indexler = np.where(y_pred != y_test)[0]
    print(f"\nToplam Hatalı Tahmin Sayısı: {len(hatali_indexler)}")
    
    if len(hatali_indexler) > 0:
        secilen_hatalar = np.random.choice(hatali_indexler, 10, replace=False)
        plt.figure(figsize=(12, 5))
        for i, index in enumerate(secilen_hatalar):
            plt.subplot(2, 5, i + 1)
            plt.imshow(X_test_img[index], cmap='gray')
            plt.title(f"Gerçek:{y_test[index]} | Tahmin:{y_pred[index]}", color='red', fontsize=10, fontweight='bold')
            plt.axis('off')
        plt.suptitle("Modelin Yanıldığı Örnekler", fontsize=16)
        plt.tight_layout()
        plt.show()

    # 4. LIME İLE AÇIKLANABİLİRLİK (XAI)
    print("LIME Analizi ")

    # Köprü Fonksiyonu: LIME (Resim) -> Model (Vektör)
    def predict_fn(images):
        # LIME renkli resim gönderir, biz griye çevirip düzleştirip scale etmeliyiz
        if images.ndim == 4 and images.shape[-1] == 3:
            gray_images = images[:, :, :, 0] # Sadece tek kanalı al
        else:
            gray_images = images
        
        flattened = gray_images.reshape(gray_images.shape[0], -1)
        scaled_input = scaler.transform(flattened)
        return clf.predict_proba(scaled_input)

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    # Görselleştirme Ayarları
    plt.style.use('dark_background')
    fig, m_axs = plt.subplots(2, 5, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    print(" -> Her rakam sınıfı (0-9) için analiz yapılıyor (Biraz zaman alabilir)...")

    for i, c_ax in enumerate(m_axs.flatten()):
        # İlgili rakam sınıfına ait ilk örneği bul
        indices = np.where(y_test == i)[0]
        if len(indices) == 0: continue
        idx = indices[0]

        # Görüntüyü LIME için hazırla (RGB formatı şart)
        image_input = X_test_img[idx]
        image_rgb = np.stack((image_input,) * 3, axis=-1)

        # Açıklamayı Hesapla
        explanation = explainer.explain_instance(
            image_rgb.astype('double'),
            predict_fn,
            labels=(i,),
            top_labels=None,
            hide_color=0,
            num_samples=1000,
            segmentation_fn=segmenter
        )

        # Maskeyi al
        temp, mask = explanation.get_image_and_mask(
            i, positive_only=True, num_features=10, hide_rest=False
        )

        # Çizim
        gorsel = mark_boundaries(temp / 255.0, mask, color=(1, 0.8, 0), mode='thick')
        c_ax.imshow(gorsel)
        c_ax.set_title(f"Rakam: {i}", fontsize=14, color='white', fontweight='bold')
        c_ax.axis('off')

    plt.suptitle("LIME Analizi: Modelin Odaklandığı Bölgeler (Sarı Çizgiler)", 
                    fontsize=18, color='cyan', y=0.98)
    plt.show()
    plt.style.use('default') # Temayı normale döndür

    # 5. MODELİ KAYDETME
    joblib.dump(clf, 'mnist_rf_model.pkl')
    joblib.dump(scaler, 'mnist_scaler.pkl')
    print(" -> KAYIT BAŞARILI: 'mnist_rf_model.pkl' ve 'mnist_scaler.pkl' oluşturuldu.")
main()