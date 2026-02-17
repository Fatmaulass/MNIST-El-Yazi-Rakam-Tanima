# ✍️ MNIST El Yazısı Rakam Tanıma: MLP ve Random Forest Analizi
Bu proje, el yazısı rakamların (0-9) bilgisayar destekli sistemler tarafından yüksek doğrulukla tanınmasını sağlayan yapay zeka modellerinin geliştirilmesini ve karşılaştırılmasını konu almaktadır. Proje kapsamında, hem modern Yapay Sinir Ağları (MLP) hem de geleneksel Random Forest yöntemleri kullanılarak kapsamlı bir analiz sunulmuştur.

🎯 Projenin Amacı
- Projenin temel hedefi, makine öğrenmesi ve görüntü işleme literatürünün standart veri seti olan MNIST üzerinde, daha önce görülmemiş verileri yüksek başarıyla sınıflandırabilen modeller eğitmektir. Çalışma, ham piksel verilerinden anlamlı öznitelik çıkarımı yapılabileceğini kanıtlar niteliktedir.

📊 Kullanılan Veri Seti (MNIST)
- Toplam Görüntü: 70.000 adet gri tonlamalı el yazısı rakam.
- Boyut: 28x28 piksel.
- Dağılım: 60.000 eğitim, 10.000 test görüntüsü.
- Ön İşleme: Piksel değerleri 0-1 aralığına normalize edilmiştir.

🏗️ Model Mimarisi (MLP), 
     Geliştirilen Çok Katmanlı Algılayıcı (MLP) modeli şu yapıya sahiptir:
- Giriş Katmanı: 784 nöron (Flattened 28x28).
- Gizli Katman: 512 nöron, ReLU aktivasyon fonksiyonu.
- Düzenlileştirme: Aşırı öğrenmeyi (overfitting) engellemek için %25 Dropout.
- Çıkış Katmanı: 10 sınıf (Logits).

🔍 Açıklanabilirlik (LIME Analizi) 
- Random Forest modeli üzerinde uygulanan LIME (Local Interpretable Model-agnostic Explanations) analizi ile modelin hangi piksellere odaklanarak karar verdiği görselleştirilmiştir. Analizler, modelin rastgele gürültüler yerine rakamın yapısal hatlarını oluşturan anlamlı bölgelere odaklandığını doğrulamıştır.

🛠️ Kullanılan Teknolojiler
- Programlama Dili: Python.
- Kütüphaneler: PyTorch, Scikit-learn, NumPy, Matplotlib, Seaborn, TorchVision.
- Açıklanabilirlik: LIME.

👩‍💻 Geliştiriciler
- Fatma Ulaş
- Sena Altıparmak
